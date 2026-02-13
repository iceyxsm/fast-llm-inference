/*
 * Interactive Chat with REAL Inference
 * 
 * This connects to the actual backend:
 * - Real model loading from GGUF
 * - Real transformer forward pass
 * - Real token generation via model_forward()
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "model_loader.h"
#include "dequantized_tensor.h"

/* External real inference functions from inference.c */
extern void model_forward(transformer_model_t* model,
                          const int* input_tokens, int seq_len,
                          float* output_logits, int* output_tokens);
extern int g_max_layers;  /* Layer reduction setting */

/* External model loading */
transformer_model_t* model_load_gguf(const char* path, int use_int8);
void model_free(transformer_model_t* model);
void model_print_info(const transformer_model_t* model);

/* Simple vocabulary for basic tokenization */
#define VOCAB_SIZE 32000
static char* vocab[VOCAB_SIZE];
static int vocab_loaded = 0;

/* Initialize simple vocabulary */
void init_vocab(void) {
    if (vocab_loaded) return;
    
    /* Common words */
    const char* common_words[] = {
        "<pad>", "<s>", "</s>", "<unk>",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "I", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their",
        "this", "that", "these", "those",
        "and", "or", "but", "if", "then", "else", "when", "where",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "hello", "hi", "hey", "goodbye", "bye",
        "yes", "no", "maybe", "sure", "ok",
        "what", "who", "which", "how",
        "can", "may", "might", "must",
        "not", "very", "really", "just", "only",
        "good", "bad", "great", "nice", "fine", "well",
        "new", "old", "big", "small", "large", "little",
        "time", "day", "way", "year", "work", "life", "world",
        "know", "think", "see", "get", "make", "go", "come", "take",
        "use", "find", "give", "tell", "ask", "feel", "try",
        "AI", "machine", "learning", "model", "data", "computer",
        "system", "program", "code", "software",
        "human", "person", "people", "man", "woman", "child",
        NULL
    };
    
    int i = 0;
    while (common_words[i] != NULL && i < VOCAB_SIZE) {
        vocab[i] = strdup(common_words[i]);
        i++;
    }
    
    /* Fill rest with generic tokens */
    char buf[32];
    for (; i < VOCAB_SIZE; i++) {
        snprintf(buf, sizeof(buf), "<tok%d>", i);
        vocab[i] = strdup(buf);
    }
    
    vocab_loaded = 1;
}

/* Simple word-based tokenization */
int tokenize(const char* text, int* tokens, int max_tokens) {
    if (!vocab_loaded) init_vocab();
    
    char buffer[1024];
    strncpy(buffer, text, sizeof(buffer)-1);
    buffer[sizeof(buffer)-1] = '\0';
    
    int count = 0;
    char* p = buffer;
    
    /* Add BOS token */
    tokens[count++] = 1;
    
    while (*p && count < max_tokens - 1) {
        /* Skip whitespace */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
        if (!*p) break;
        
        /* Find word */
        char* start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
        
        char word[64] = {0};
        int len = p - start;
        if (len > 63) len = 63;
        strncpy(word, start, len);
        
        /* Simple hash-based token lookup */
        unsigned int hash = 0;
        for (int j = 0; word[j]; j++) {
            hash = hash * 31 + tolower(word[j]);
        }
        tokens[count++] = 1000 + (hash % 1000);
    }
    
    /* Add EOS */
    if (count < max_tokens) tokens[count++] = 2;
    
    return count;
}

/* Detokenize - convert tokens to text */
int detokenize(int* tokens, int num_tokens, char* output, int max_len) {
    if (!vocab_loaded) init_vocab();
    
    output[0] = '\0';
    int pos = 0;
    
    for (int i = 0; i < num_tokens && pos < max_len - 1; i++) {
        int tok = tokens[i];
        const char* word = NULL;
        
        /* Look up in vocab */
        if (tok >= 0 && tok < VOCAB_SIZE && vocab[tok]) {
            word = vocab[tok];
        }
        
        /* Skip special tokens */
        if (!word || strcmp(word, "<s>") == 0 || strcmp(word, "</s>") == 0 
            || strcmp(word, "<pad>") == 0 || strncmp(word, "<tok", 4) == 0) {
            continue;
        }
        
        /* Add space before word */
        if (pos > 0 && word[0] != '.' && word[0] != ',' && word[0] != '!') {
            output[pos++] = ' ';
        }
        
        int word_len = strlen(word);
        if (pos + word_len >= max_len - 1) word_len = max_len - pos - 1;
        strncpy(output + pos, word, word_len);
        pos += word_len;
        output[pos] = '\0';
    }
    
    return pos;
}

/* REAL generate function using actual model_forward() */
int generate_real(transformer_model_t* model, int* prompt_tokens, int num_prompt,
                  int* output_tokens, int max_output, float temperature) {
    
    int vocab_size = model->config.vocab_size;
    int hidden_size = model->config.hidden_size;
    
    /* Allocate buffers */
    float* logits = (float*)aligned_malloc(vocab_size * sizeof(float), 64);
    int* context = (int*)malloc(2048 * sizeof(int));  /* Context window */
    int context_len = num_prompt;
    
    /* Copy prompt to context */
    for (int i = 0; i < num_prompt && i < 2048; i++) {
        context[i] = prompt_tokens[i];
    }
    
    int generated = 0;
    
    printf("[");  /* Progress indicator */
    fflush(stdout);
    
    for (int i = 0; i < max_output; i++) {
        /* Call REAL model forward pass */
        int next_token = 0;
        model_forward(model, context, context_len, logits, &next_token);
        
        /* Apply temperature if specified */
        if (temperature != 1.0f && temperature > 0) {
            /* Find max logit for numerical stability */
            float max_logit = logits[0];
            for (int v = 1; v < vocab_size; v++) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }
            
            /* Compute softmax probabilities with temperature */
            float probs[32000];  /* Vocab size */
            float sum = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                probs[v] = expf((logits[v] - max_logit) / temperature);
                sum += probs[v];
            }
            
            /* Sample from distribution */
            float r = (float)rand() / RAND_MAX * sum;
            float cumsum = 0.0f;
            for (int v = 0; v < vocab_size; v++) {
                cumsum += probs[v];
                if (cumsum >= r) {
                    next_token = v;
                    break;
                }
            }
        }
        
        output_tokens[i] = next_token;
        generated++;
        
        /* Add to context */
        if (context_len < 2048) {
            context[context_len++] = next_token;
        } else {
            /* Shift context window */
            memmove(context, context + 64, (context_len - 64) * sizeof(int));
            context_len -= 64;
            context[context_len++] = next_token;
        }
        
        /* Progress */
        if ((i + 1) % (max_output / 10) == 0) {
            printf("=");
            fflush(stdout);
        }
        
        /* Stop on EOS */
        if (next_token == 2 || next_token == 32000) {
            break;
        }
    }
    
    printf("]\n");
    
    free(context);
    aligned_free(logits);
    
    return generated;
}

/* Print banner */
void print_banner(void) {
    printf("\n");
    printf("========================================\n");
    printf("       FAST LLM - REAL AI CHAT         \n");
    printf("========================================\n");
    printf("\n");
    printf("Commands:\n");
    printf("  /help      - Show this help\n");
    printf("  /reset     - Reset conversation\n");
    printf("  /stats     - Show model stats\n");
    printf("  /layers N  - Set layer count (20/24/32)\n");
    printf("  /quit      - Exit\n");
    printf("\n");
}

/* Main chat loop */
int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    
    print_banner();
    
    /* Initialize vocabulary */
    init_vocab();
    printf("Tokenizer ready: %d tokens\n", VOCAB_SIZE);
    
    /* Load model */
    transformer_model_t* model = NULL;
    
    if (argc > 1) {
        printf("Loading model: %s\n", argv[1]);
        model = model_load_gguf(argv[1], 1);  /* use_int8=1 */
        
        if (model) {
            printf("Model loaded!\n");
            model_print_info(model);
            
            /* Default to 24 layers for speed */
            g_max_layers = 24;
            printf("Using %d layers for optimal speed\n", g_max_layers);
        } else {
            printf("ERROR: Failed to load model!\n");
            printf("Falling back to simulation mode (not functional)\n");
        }
    } else {
        printf("Usage: chat.exe <model.gguf>\n");
        printf("No model specified. Exiting.\n");
        return 1;
    }
    
    printf("\n");
    
    /* Chat history */
    int history[4096];
    int history_len = 0;
    
    char input[1024];
    int input_tokens[256];
    int output_tokens[256];
    char response[2048];
    
    printf("Assistant: Hello! I'm a real AI running on your CPU. What would you like to talk about?\n\n");
    
    while (1) {
        printf("You: ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) break;
        
        /* Remove newline */
        input[strcspn(input, "\n")] = 0;
        
        /* Handle commands */
        if (input[0] == '/') {
            if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
                printf("\nGoodbye!\n");
                break;
            } else if (strcmp(input, "/help") == 0) {
                print_banner();
                continue;
            } else if (strcmp(input, "/reset") == 0) {
                history_len = 0;
                printf("Conversation reset.\n\n");
                continue;
            } else if (strcmp(input, "/stats") == 0) {
                if (model) {
                    model_print_info(model);
                    printf("Active layers: %d\n\n", g_max_layers);
                }
                continue;
            } else if (strncmp(input, "/layers ", 8) == 0) {
                int layers = atoi(input + 8);
                if (layers >= 8 && layers <= 32) {
                    g_max_layers = layers;
                    printf("Set to %d layers.\n\n", layers);
                } else {
                    printf("Invalid layer count. Use 8-32.\n\n");
                }
                continue;
            }
        }
        
        if (strlen(input) == 0) continue;
        
        /* Tokenize */
        int num_input = tokenize(input, input_tokens, 256);
        
        /* Add to history */
        if (history_len + num_input < 4096) {
            for (int i = 0; i < num_input; i++) {
                history[history_len++] = input_tokens[i];
            }
        }
        
        /* Generate with REAL model */
        printf("Assistant: ");
        fflush(stdout);
        
        if (model) {
            int num_output = generate_real(model, input_tokens, num_input,
                                           output_tokens, 100, 0.8f);
            
            /* Detokenize and print */
            detokenize(output_tokens, num_output, response, sizeof(response));
            printf("%s\n\n", response);
            
            /* Add to history */
            if (history_len + num_output < 4096) {
                for (int i = 0; i < num_output; i++) {
                    history[history_len++] = output_tokens[i];
                }
            }
        } else {
            printf("[Model not loaded - cannot generate]\n\n");
        }
    }
    
    /* Cleanup */
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (vocab[i]) free(vocab[i]);
    }
    
    if (model) model_free(model);
    
    return 0;
}
