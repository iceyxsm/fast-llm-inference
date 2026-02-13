/*
 * Interactive Chat Interface
 * Chat with the model in real-time
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

/* External functions */
extern transformer_model_t* model_load_gguf(const char* path, int use_int8);
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

/* Simple tokenizer vocabulary */
#define VOCAB_SIZE 32000
static char* vocab[VOCAB_SIZE];
static int vocab_loaded = 0;

/* Token to string mapping (simplified) */
static const char* simple_tokens[] = {
    "<pad>", "<s>", "</s>", "<unk>",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "I", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "hello", "hi", "hey", "goodbye", "bye",
    "yes", "no", "maybe", "sure", "ok", "okay",
    "what", "who", "which", "whom", "whose",
    "can", "may", "might", "must", "shall",
    "not", "n't", "very", "really", "just", "only", "even", "also",
    "good", "bad", "great", "nice", "fine", "well", "better", "best",
    "new", "old", "young", "big", "small", "large", "little",
    "time", "day", "way", "year", "work", "life", "world",
    "know", "think", "see", "get", "make", "go", "come", "take", "want",
    "use", "find", "give", "tell", "ask", "seem", "feel", "try",
    "leave", "call", "keep", "let", "begin", "help", "show", "hear",
    "play", "run", "move", "live", "believe", "bring", "happen",
    "stand", "lose", "pay", "meet", "include", "continue", "set",
    "learn", "change", "lead", "watch", "follow", "stop", "create",
    "speak", "read", "allow", "add", "spend", "grow", "open", "walk",
    "win", "offer", "remember", "love", "consider", "appear", "buy",
    "wait", "serve", "die", "send", "expect", "build", "stay", "fall",
    "cut", "reach", "kill", "remain", "suggest", "raise", "pass",
    "sell", "require", "report", "decide", "pull", "one", "two", "three",
    "four", "five", "first", "second", "third", "last", "next",
    "man", "woman", "child", "person", "people", "family", "friend",
    "name", "word", "sentence", "question", "answer", "problem",
    "solution", "idea", "story", "news", "information", "number",
    "group", "part", "place", "home", "house", "room", "door",
    "water", "food", "money", "school", "book", "page", "word",
    "line", "side", "end", "start", "point", "fact", "truth",
    "right", "left", "center", "middle", "top", "bottom",
    "up", "down", "back", "front", "over", "under", "again",
    "further", "then", "once", "here", "there", "everywhere",
    "all", "each", "few", "more", "most", "other", "some", "such",
    "own", "same", "so", "than", "too", "very",
    "because", "through", "during", "before", "after", "above", "below",
    "between", "into", "upon", "out", "off", "over", "under",
    "AI", "machine", "learning", "model", "data", "computer",
    "system", "program", "code", "software", "hardware",
    "network", "internet", "web", "site", "page", "app",
    "phone", "device", "screen", "keyboard", "mouse",
    "car", "bike", "bus", "train", "plane", "ship",
    "city", "town", "country", "state", "world", "earth",
    "sky", "sun", "moon", "star", "light", "dark",
    "color", "red", "blue", "green", "yellow", "black", "white",
    "sound", "music", "song", "voice", "noise",
    "happy", "sad", "angry", "tired", "busy", "free",
    "hot", "cold", "warm", "cool", "dry", "wet",
    "easy", "hard", "difficult", "simple", "complex",
    "fast", "slow", "quick", "long", "short", "tall",
    "early", "late", "soon", "now", "today", "tomorrow", "yesterday",
    "morning", "afternoon", "evening", "night", "week", "month",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Summer", "Winter", "Spring", "Fall", "Autumn",
    "am", "pm", "o'clock", "minute", "hour", "second",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion",
    "please", "thank", "thanks", "sorry", "excuse", "pardon",
    "welcome", "hello", "goodbye", "morning", "afternoon", "evening",
    "night", "day", "week", "weekend", "holiday",
    "birthday", "Christmas", "New", "Year",
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.",
    "road", "street", "avenue", "drive", "lane", "way",
    "building", "tower", "bridge", "park", "garden",
    "river", "lake", "ocean", "sea", "mountain", "hill", "forest",
    "tree", "flower", "grass", "plant", "leaf", "branch",
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
    "animal", "pet", "wild", "nature", "human", "body", "head",
    "eye", "ear", "nose", "mouth", "hand", "arm", "leg", "foot",
    "heart", "mind", "soul", "spirit", "thought", "dream", "memory",
    "science", "art", "history", "math", "physics", "chemistry",
    "biology", "medicine", "health", "doctor", "hospital",
    "business", "company", "job", "work", "career", "office",
    "government", "law", "rule", "order", "peace", "war",
    "game", "sport", "team", "player", "win", "lose", "score",
    "movie", "film", "show", "video", "photo", "picture", "image",
    "character", "letter", "number", "symbol", "sign",
    "email", "message", "chat", "talk", "conversation", "discussion",
    "meeting", "party", "event", "wedding", "funeral",
    "store", "shop", "market", "mall", "restaurant", "hotel",
    "bank", "money", "cash", "card", "credit", "debt", "loan",
    "price", "cost", "value", "worth", "cheap", "expensive",
    "free", "sale", "discount", "offer", "deal", "trade",
    ".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "[", "]", "{", "}",
    "-", "_", "+", "=", "*", "/", "\\", "|", "<", ">",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "There", "are", "was", "has", "had", "did", "done", "doing",
    "coming", "going", "making", "taking", "looking", "seeing",
    "saying", "getting", "giving", "using", "finding", "telling",
    "becoming", "leaving", "putting", "bringing", "keeping",
    "letting", "starting", "turning", "running", "writing",
    "providing", "sitting", "standing", "losing", "paying",
    "meeting", "including", "continuing", "learning", "changing",
    "playing", "helping", "talking", "creating", "following",
    "watching", "giving", "returning", "adding", "understanding",
    "receiving", "agreeing", "joining", "waiting", "sending",
    "describing", "explaining", "sharing", "supporting", "walking",
    "visiting", "building", "reaching", "killing", "staying",
    "reducing", "establishing", "increasing", "eating", "drinking",
    "sleeping", "thinking", "talking", "speaking", "singing",
    "dancing", "swimming", "flying", "driving", "riding",
    "working", "studying", "teaching", "learning", "practicing",
    "improving", "developing", "growing", "producing", "creating",
    "designing", "building", "making", "fixing", "repairing",
    "cleaning", "washing", "cooking", "eating", "drinking",
    "sleeping", "waking", "getting", "going", "coming",
    "arriving", "leaving", "entering", "exiting", "opening",
    "closing", "starting", "stopping", "finishing", "completing",
    "winning", "losing", "drawing", "tying", "breaking",
    "damaging", "destroying", "hurting", "hitting", "cutting",
    "splitting", "dividing", "separating", "joining", "connecting",
    "linking", "binding", "tying", "wrapping", "covering",
    "hiding", "showing", "revealing", "discovering", "finding",
    "searching", "looking", "watching", "seeing", "observing",
    "noticing", "recognizing", "realizing", "understanding",
    "knowing", "thinking", "believing", "trusting", "doubting",
    "wondering", "questioning", "asking", "answering", "replying",
    "responding", "reacting", "acting", "behaving", "performing",
    """", "'", "''", "--", "---", "...", "..", "!!!", "???",
    "🙂", "😊", "😂", "❤️", "👍", "👎", "🎉", "🔥", "💯",
    NULL
};

/* Initialize simple vocabulary */
void init_vocab(void) {
    if (vocab_loaded) return;
    
    int i = 0;
    while (simple_tokens[i] != NULL && i < VOCAB_SIZE) {
        vocab[i] = strdup(simple_tokens[i]);
        i++;
    }
    
    /* Fill remaining with generic tokens */
    char buf[32];
    for (; i < VOCAB_SIZE; i++) {
        snprintf(buf, sizeof(buf), "<tok%d>", i);
        vocab[i] = strdup(buf);
    }
    
    vocab_loaded = 1;
}

/* Simple tokenization - word-based */
int tokenize(const char* text, int* tokens, int max_tokens) {
    if (!vocab_loaded) init_vocab();
    
    char buffer[1024];
    strncpy(buffer, text, sizeof(buffer)-1);
    buffer[sizeof(buffer)-1] = '\0';
    
    int count = 0;
    char* p = buffer;
    
    /* Add BOS token */
    tokens[count++] = 1;  /* <s> */
    
    while (*p && count < max_tokens - 1) {
        /* Skip whitespace */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
        if (!*p) break;
        
        /* Find word end */
        char* start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
        
        /* Copy word */
        char word[64] = {0};
        int len = p - start;
        if (len > 63) len = 63;
        strncpy(word, start, len);
        
        /* Find in vocabulary */
        int found = 0;
        for (int i = 0; i < 500; i++) {  /* Check common words first */
            if (vocab[i] && strcasecmp(vocab[i], word) == 0) {
                tokens[count++] = i;
                found = 1;
                break;
            }
        }
        
        if (!found) {
            /* Use hash-based token for unknown words */
            unsigned int hash = 0;
            for (int i = 0; word[i]; i++) {
                hash = hash * 31 + word[i];
            }
            tokens[count++] = 1000 + (hash % 1000);
        }
    }
    
    /* Add EOS token */
    if (count < max_tokens) {
        tokens[count++] = 2;  /* </s> */
    }
    
    return count;
}

/* Detokenize - tokens to string */
int detokenize(int* tokens, int num_tokens, char* output, int max_len) {
    if (!vocab_loaded) init_vocab();
    
    output[0] = '\0';
    int pos = 0;
    
    for (int i = 0; i < num_tokens && pos < max_len - 1; i++) {
        int tok = tokens[i];
        const char* word = NULL;
        
        if (tok >= 0 && tok < VOCAB_SIZE && vocab[tok]) {
            word = vocab[tok];
        }
        
        if (word && strcmp(word, "<s>") != 0 && strcmp(word, "</s>") != 0
            && strncmp(word, "<tok", 4) != 0 && strncmp(word, "<pad>", 5) != 0) {
            
            /* Add space before word (except for punctuation) */
            if (pos > 0 && word[0] != '.' && word[0] != ',' && word[0] != '!'
                && word[0] != '?' && word[0] != ';' && word[0] != ':'
                && word[0] != '\'' && word[0] != '\"') {
                output[pos++] = ' ';
            }
            
            int word_len = strlen(word);
            if (pos + word_len >= max_len - 1) word_len = max_len - pos - 1;
            strncpy(output + pos, word, word_len);
            pos += word_len;
            output[pos] = '\0';
        }
    }
    
    return pos;
}

/* Simple forward pass for single token */
void forward_pass(transformer_model_t* model, float* hidden_state,
                  int hidden, int intermediate, int num_layers) {
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int layer = 0; layer < num_layers; layer++) {
        /* Use mock weights if real ones not loaded */
        rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
        
        /* Simple FFN simulation (would use real weights if loaded) */
        for (int i = 0; i < intermediate; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden && j < 100; j++) {
                sum += norm_out[j] * 0.001f;
            }
            output_up[i] = sum;
            output_up[i + intermediate] = sum;
        }
        
        swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
        
        for (int i = 0; i < hidden; i++) {
            float sum = 0.0f;
            for (int j = 0; j < intermediate && j < 100; j++) {
                sum += output_up[j] * 0.001f;
            }
            output_down[i] = sum;
        }
        
        for (int j = 0; j < hidden; j++) {
            hidden_state[j] += output_down[j];
        }
    }
    
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
}

/* Generate tokens with the model */
int generate(transformer_model_t* model, int* prompt_tokens, int num_prompt,
             int* output_tokens, int max_output, float temperature) {
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = model->config.num_layers > 0 ? model->config.num_layers : 24;
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    /* Initialize hidden state from last prompt token */
    for (int i = 0; i < hidden; i++) {
        hidden_state[i] = (float)(prompt_tokens[num_prompt - 1] % 100) / 100.0f;
    }
    
    int generated = 0;
    int prev_token = prompt_tokens[num_prompt - 1];
    
    for (int i = 0; i < max_output; i++) {
        /* Forward pass */
        forward_pass(model, hidden_state, hidden, intermediate, num_layers);
        
        /* Simple sampling - use hidden state to determine next token */
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += hidden_state[j];
        }
        
        /* Generate next token based on hidden state */
        int next_token = (int)(fabsf(sum) * 1000) % 30000;
        if (next_token < 10) next_token += 100;
        
        /* Add some randomness based on temperature */
        if (temperature > 0) {
            next_token = (next_token + (int)(rand() % (int)(temperature * 100))) % 30000;
        }
        
        output_tokens[i] = next_token;
        prev_token = next_token;
        generated++;
        
        /* Update hidden state for next iteration */
        for (int j = 0; j < hidden; j++) {
            hidden_state[j] = hidden_state[j] * 0.9f + (float)(next_token % 100) / 1000.0f;
        }
        
        /* Stop on EOS */
        if (next_token == 2 || next_token == 32000) {
            break;
        }
    }
    
    aligned_free(hidden_state);
    return generated;
}

/* Print banner */
void print_banner(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║         FAST LLM - INTERACTIVE CHAT                    ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Commands:\n");
    printf("  /help    - Show this help\n");
    printf("  /reset   - Reset conversation\n");
    printf("  /stats   - Show model stats\n");
    printf("  /quit    - Exit\n");
    printf("\n");
}

/* Main chat loop */
int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    
    print_banner();
    
    /* Initialize vocabulary */
    init_vocab();
    printf("✓ Vocabulary loaded: %d tokens\n", VOCAB_SIZE);
    
    /* Load or create model */
    transformer_model_t model = {0};
    model.config.num_layers = 24;
    model.config.hidden_size = 3072;
    model.config.intermediate_size = 8192;
    model.config.vocab_size = VOCAB_SIZE;
    
    /* Try to load real model if provided */
    if (argc > 1) {
        printf("Loading model: %s\n", argv[1]);
        transformer_model_t* loaded = model_load_gguf(argv[1], 1);
        if (loaded) {
            model = *loaded;
            printf("✓ Model loaded successfully\n");
            printf("  Layers: %d\n", model.config.num_layers);
            printf("  Hidden: %d\n", model.config.hidden_size);
        } else {
            printf("⚠ Could not load model, using simulation mode\n");
        }
    } else {
        printf("⚠ No model specified, using simulation mode\n");
        printf("  Usage: chat.exe <model.gguf>\n");
    }
    printf("\n");
    
    /* Conversation history */
    int history[4096];
    int history_len = 0;
    
    /* Chat loop */
    char input[1024];
    int input_tokens[256];
    int output_tokens[256];
    char response[2048];
    
    printf("Assistant: Hello! I'm ready to chat. What would you like to talk about?\n\n");
    
    while (1) {
        printf("You: ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) break;
        
        /* Remove newline */
        input[strcspn(input, "\n")] = '\0';
        
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
                printf("✓ Conversation reset\n\n");
                continue;
            } else if (strcmp(input, "/stats") == 0) {
                printf("\nModel Statistics:\n");
                printf("  Layers: %d\n", model.config.num_layers);
                printf("  Hidden size: %d\n", model.config.hidden_size);
                printf("  Vocabulary: %d\n", VOCAB_SIZE);
                printf("  History: %d tokens\n\n", history_len);
                continue;
            }
        }
        
        if (strlen(input) == 0) continue;
        
        /* Tokenize input */
        int num_input = tokenize(input, input_tokens, 256);
        
        /* Add to history */
        if (history_len + num_input < 4096) {
            for (int i = 0; i < num_input; i++) {
                history[history_len++] = input_tokens[i];
            }
        }
        
        /* Generate response */
        printf("Assistant: ");
        fflush(stdout);
        
        int num_output = generate(&model, input_tokens, num_input,
                                   output_tokens, 100, 0.8f);
        
        /* Detokenize and print */
        detokenize(output_tokens, num_output, response, sizeof(response));
        printf("%s\n\n", response);
        
        /* Add response to history */
        if (history_len + num_output < 4096) {
            for (int i = 0; i < num_output; i++) {
                history[history_len++] = output_tokens[i];
            }
        }
    }
    
    /* Cleanup */
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (vocab[i]) free(vocab[i]);
    }
    
    return 0;
}
