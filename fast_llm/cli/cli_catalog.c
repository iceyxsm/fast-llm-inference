/*
 * fllm CLI - Model catalog
 *
 * Fetches models from HuggingFace per category (LLM, Vision, Code, Voice, STT).
 * Interactive type selector → parameter filter → clean table with Yes/No caps.
 */
#include "cli.h"

/* ── JSON helpers ─────────────────────────────────────────────────── */

static int json_str(const char* pos, char* out, int sz) {
    if (!pos || *pos != '"') return 0;
    pos++;
    int i = 0;
    while (*pos && *pos != '"' && i < sz - 1) {
        if (*pos == '\\' && *(pos+1)) pos++;
        out[i++] = *pos++;
    }
    out[i] = '\0';
    return i > 0;
}

static void trim(char* s) {
    while (s[0] == ' ') memmove(s, s+1, strlen(s));
    int len = (int)strlen(s);
    while (len > 0 && (s[len-1]==' '||s[len-1]=='\r'||s[len-1]=='\n')) s[--len]='\0';
}

static char* fetch_json(const char* url, int max_bytes) {
    char tmp[1024];
    cli_get_fllm_dir(tmp, sizeof(tmp));
    strncat(tmp, PATH_SEP_S "_tmp.json", sizeof(tmp)-strlen(tmp)-1);
    char cmd[2048];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "curl.exe -sL -o \"%s\" \"%s\"", tmp, url);
#else
    snprintf(cmd, sizeof(cmd), "curl -sL -o '%s' '%s' 2>/dev/null || wget -q -O '%s' '%s' 2>/dev/null", tmp, url, tmp, url);
#endif
    system(cmd);
    if (!cli_file_exists(tmp)) return NULL;
    FILE* f = fopen(tmp, "rb");
    if (!f) { remove(tmp); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > max_bytes) { fclose(f); remove(tmp); return NULL; }
    char* buf = (char*)malloc(sz+1);
    fread(buf, 1, sz, f); buf[sz] = '\0';
    fclose(f); remove(tmp);
    return buf;
}

/* ── Type name ────────────────────────────────────────────────────── */

const char* catalog_type_name(model_type_t t) {
    switch (t) {
        case MTYPE_LLM:    return "Chat / Text";
        case MTYPE_VISION: return "Vision (Image)";
        case MTYPE_CODE:   return "Code";
        case MTYPE_VOICE:  return "Voice (TTS)";
        case MTYPE_STT:    return "Speech-to-Text";
        default:           return "Unknown";
    }
}

/* ── Detection helpers ────────────────────────────────────────────── */

static int detect_vision(const char* n) {
    return (strstr(n,"Vision")||strstr(n,"vision")||strstr(n,"VL")||strstr(n,"-vl")||strstr(n,"LLaVA")||strstr(n,"llava")||strstr(n,"Pixtral")||strstr(n,"InternVL")||strstr(n,"Qwen2-VL")||strstr(n,"Qwen2.5-VL")) ? 1 : 0;
}
static int detect_tools(const char* n) {
    return (strstr(n,"Instruct")||strstr(n,"instruct")||strstr(n,"Chat")||strstr(n,"chat")) ? 1 : 0;
}
static int detect_code(const char* n) {
    return (strstr(n,"Code")||strstr(n,"code")||strstr(n,"Coder")||strstr(n,"coder")||strstr(n,"StarCoder")||strstr(n,"CodeLlama")) ? 1 : 0;
}
static int detect_voice(const char* n) {
    return (strstr(n,"TTS")||strstr(n,"tts")||strstr(n,"VITS")||strstr(n,"vits")||strstr(n,"Bark")||strstr(n,"bark")||strstr(n,"Piper")||strstr(n,"piper")||strstr(n,"OuteTTS")||strstr(n,"Kokoro")) ? 1 : 0;
}
static int detect_stt(const char* n) {
    return (strstr(n,"Whisper")||strstr(n,"whisper")||strstr(n,"speech-to-text")||strstr(n,"Distil-Whisper")||strstr(n,"faster-whisper")) ? 1 : 0;
}

static double guess_params(const char* name) {
    const char* p = name;
    while (*p) {
        if ((*p>='0'&&*p<='9')||*p=='.') {
            double v = atof(p);
            while (*p&&((*p>='0'&&*p<='9')||*p=='.')) p++;
            if (*p=='B'||*p=='b') return v;
        }
        p++;
    }
    return 0;
}

static int guess_context(const char* n) {
    if (strstr(n,"Llama-3")) return 131072;
    if (strstr(n,"Qwen2")) return 32768;
    if (strstr(n,"Mistral")) return 32768;
    if (strstr(n,"Gemma")) return 8192;
    return 4096;
}

static void extract_quant(const char* fn, char* out, int sz) {
    const char* p[] = {"Q4_K_M","Q4_K_S","Q4_0","Q5_K_M","Q5_K_S","Q3_K_M","Q6_K","Q8_0","f16",NULL};
    for (int i=0;p[i];i++) if (strstr(fn,p[i])) { snprintf(out,sz,"%s",p[i]); return; }
    snprintf(out,sz,"%s","Q4");
}

static void extract_family(const char* n, char* out, int sz) {
    if (strstr(n,"Llama")||strstr(n,"TinyLlama")) snprintf(out,sz,"%s","Llama");
    else if (strstr(n,"Mistral")||strstr(n,"Mixtral")) snprintf(out,sz,"%s","Mistral");
    else if (strstr(n,"Qwen")) snprintf(out,sz,"%s","Qwen");
    else if (strstr(n,"Phi")) snprintf(out,sz,"%s","Phi");
    else if (strstr(n,"Gemma")) snprintf(out,sz,"%s","Gemma");
    else if (strstr(n,"DeepSeek")) snprintf(out,sz,"%s","DeepSeek");
    else if (strstr(n,"Whisper")) snprintf(out,sz,"%s","Whisper");
    else snprintf(out,sz,"%s","Other");
}

static void make_short_name(const char* full, char* out, int sz) {
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "%s", full);

    /* Remove common suffixes */
    char* suf[] = {"-GGUF","_GGUF","-Instruct","-instruct","-Chat","-chat","-it",
                   "-v0.1","-v0.2","-v0.3","-v1","-v2","-v3","-V1","-V2","-V3",
                   "-ft-awesome-chatgpt-prompts-gguf", NULL};
    for (int i=0;suf[i];i++) { char* p=strstr(tmp,suf[i]); if(p)*p='\0'; }

    /* Replace hyphens/underscores with spaces */
    for (char* p=tmp;*p;p++) if(*p=='-'||*p=='_') *p=' ';

    /* Remove parameter size tokens like "0.5B", "1B", "3B", "7B", "8B", "24B", "A4B" etc.
       Scan words and drop any that match [number]B or A[number]B pattern */
    char cleaned[128] = {0};
    int cpos = 0;
    char* tok = strtok(tmp, " ");
    while (tok) {
        int is_param = 0;
        int len = (int)strlen(tok);

        /* Check patterns: "0.5B", "1B", "3B", "7B", "8B", "24B", "31B" */
        if (len >= 2 && (tok[len-1]=='B' || tok[len-1]=='b')) {
            /* Check if everything before B is digits/dots */
            int all_num = 1;
            for (int i=0; i<len-1; i++) {
                if (!((tok[i]>='0'&&tok[i]<='9')||tok[i]=='.')) { all_num=0; break; }
            }
            if (all_num) is_param = 1;
        }

        /* "A4B", "A16B" pattern */
        if (len >= 3 && (tok[0]=='A'||tok[0]=='a') && (tok[len-1]=='B'||tok[len-1]=='b')) {
            int all_num = 1;
            for (int i=1; i<len-1; i++) {
                if (!(tok[i]>='0'&&tok[i]<='9')) { all_num=0; break; }
            }
            if (all_num) is_param = 1;
        }

        /* Also skip "4k", "128k" context tokens and "Q4", "Q4_K_M" quant tokens */
        if (len >= 2 && (tok[len-1]=='K'||tok[len-1]=='k')) {
            int all_num = 1;
            for (int i=0; i<len-1; i++) {
                if (!(tok[i]>='0'&&tok[i]<='9')) { all_num=0; break; }
            }
            if (all_num) is_param = 1;
        }
        if (tok[0]=='Q' && len>=2 && (tok[1]>='0'&&tok[1]<='9')) is_param = 1;

        /* Skip "Meta" prefix — just noise */
        if (strcmp(tok,"Meta")==0 || strcmp(tok,"meta")==0) is_param = 1;

        if (!is_param) {
            if (cpos > 0 && cpos < (int)sizeof(cleaned)-1) cleaned[cpos++] = ' ';
            int remain = (int)sizeof(cleaned) - cpos - 1;
            if (remain > 0) {
                int copy = len < remain ? len : remain;
                memcpy(cleaned+cpos, tok, copy);
                cpos += copy;
            }
        }
        tok = strtok(NULL, " ");
    }
    cleaned[cpos] = '\0';
    trim(cleaned);
    snprintf(out, sz, "%s", cleaned);
}

static hw_tier_t tier_from_params(double p) {
    if (p>=7.0) return TIER_HIGH;
    if (p>=3.0) return TIER_MEDIUM;
    return TIER_LOW;
}

static model_type_t classify_model(const char* name) {
    if (detect_stt(name)) return MTYPE_STT;
    if (detect_voice(name)) return MTYPE_VOICE;
    if (detect_vision(name)) return MTYPE_VISION;
    if (detect_code(name)) return MTYPE_CODE;
    return MTYPE_LLM;
}

/* ── Search queries per model type ────────────────────────────────── */

static const char** get_queries(model_type_t type) {
    static const char* LLM_Q[] = {
        "https://huggingface.co/api/models?search=Llama+GGUF+Instruct&sort=downloads&direction=-1&limit=8&filter=gguf",
        "https://huggingface.co/api/models?search=Qwen2.5+GGUF+Instruct&sort=downloads&direction=-1&limit=6&filter=gguf",
        "https://huggingface.co/api/models?search=Mistral+GGUF+Instruct&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=Phi+GGUF+Instruct&sort=downloads&direction=-1&limit=4&filter=gguf",
        "https://huggingface.co/api/models?search=Gemma+GGUF+it&sort=downloads&direction=-1&limit=4&filter=gguf",
        "https://huggingface.co/api/models?search=TinyLlama+GGUF&sort=downloads&direction=-1&limit=2&filter=gguf",
        NULL
    };
    static const char* VISION_Q[] = {
        "https://huggingface.co/api/models?search=LLaVA+GGUF&sort=downloads&direction=-1&limit=8&filter=gguf",
        "https://huggingface.co/api/models?search=Qwen2-VL+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=Pixtral+GGUF&sort=downloads&direction=-1&limit=4&filter=gguf",
        "https://huggingface.co/api/models?search=InternVL+GGUF&sort=downloads&direction=-1&limit=4&filter=gguf",
        NULL
    };
    static const char* CODE_Q[] = {
        "https://huggingface.co/api/models?search=Qwen2.5-Coder+GGUF&sort=downloads&direction=-1&limit=6&filter=gguf",
        "https://huggingface.co/api/models?search=DeepSeek-Coder+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=CodeLlama+GGUF&sort=downloads&direction=-1&limit=4&filter=gguf",
        "https://huggingface.co/api/models?search=StarCoder+GGUF&sort=downloads&direction=-1&limit=3&filter=gguf",
        NULL
    };
    static const char* VOICE_Q[] = {
        "https://huggingface.co/api/models?search=OuteTTS+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=Kokoro+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=Piper+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=VITS+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        "https://huggingface.co/api/models?search=TTS+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        NULL
    };
    static const char* STT_Q[] = {
        "https://huggingface.co/api/models?search=Whisper+GGUF&sort=downloads&direction=-1&limit=10&filter=gguf",
        "https://huggingface.co/api/models?search=Distil-Whisper+GGUF&sort=downloads&direction=-1&limit=5&filter=gguf",
        NULL
    };

    switch (type) {
        case MTYPE_LLM:    return LLM_Q;
        case MTYPE_VISION: return VISION_Q;
        case MTYPE_CODE:   return CODE_Q;
        case MTYPE_VOICE:  return VOICE_Q;
        case MTYPE_STT:    return STT_Q;
        default:           return LLM_Q;
    }
}

/* ── Parse one repo ───────────────────────────────────────────────── */

static int parse_repo(const char* repo_id, model_entry_t* e) {
    memset(e, 0, sizeof(*e));
    char url[512];
    snprintf(url, sizeof(url), "https://huggingface.co/api/models/%s", repo_id);
    char* json = fetch_json(url, 2*1024*1024);
    if (!json) return 0;

    const char* sib = strstr(json, "\"siblings\"");
    if (!sib) { free(json); return 0; }

    char best_file[256]={0}; long best_size=0; int best_prio=0;
    const char* cur = sib;
    while ((cur = strstr(cur, "\"rfilename\"")) != NULL) {
        cur = strchr(cur, ':'); if (!cur) break; cur++;
        while (*cur==' ') cur++;
        char fn[256]={0};
        if (!json_str(cur, fn, sizeof(fn))) { cur++; continue; }
        if (!strstr(fn, ".gguf")) { cur++; continue; }
        if (strstr(fn, "-00001")||strstr(fn, ".part")) { cur++; continue; }

        int p=1;
        if (strstr(fn,"Q4_K_M")) p=10;
        else if (strstr(fn,"Q4_0")) p=8;
        else if (strstr(fn,"Q4_K_S")) p=7;
        else if (strstr(fn,"Q5_K_M")) p=6;
        else if (strstr(fn,"Q3_K_M")) p=5;
        else if (strstr(fn,"Q8_0")) p=3;
        else if (strstr(fn,"Q6_K")) p=3;

        const char* sz_p = strstr(cur, "\"size\"");
        long fsz=0;
        if (sz_p) { sz_p=strchr(sz_p,':'); if(sz_p) fsz=(long)atof(sz_p+1); }

        if (p > best_prio) { snprintf(best_file,sizeof(best_file),"%s",fn); best_size=fsz; best_prio=p; }
        cur++;
    }
    if (best_file[0]=='\0') { free(json); return 0; }

    const char* dl_p = strstr(json, "\"downloads\"");
    if (dl_p) { dl_p=strchr(dl_p,':'); if(dl_p) e->downloads=(int)atof(dl_p+1); }
    const char* lic_p = strstr(json, "\"license\"");
    if (lic_p) { lic_p=strchr(lic_p,':'); if(lic_p) { lic_p++; while(*lic_p==' ')lic_p++; json_str(lic_p,e->license,sizeof(e->license)); } }

    free(json);

    snprintf(e->id, sizeof(e->id), "%s", repo_id);
    snprintf(e->filename, sizeof(e->filename), "%s", best_file);
    snprintf(e->url, sizeof(e->url), "https://huggingface.co/%s/resolve/main/%s", repo_id, best_file);

    char clean[128];
    snprintf(clean, sizeof(clean), "%s", repo_id);
    char* sl=strchr(clean,'/'); if(sl) memmove(clean,sl+1,strlen(sl));
    char* gf=strstr(clean,"-GGUF"); if(gf)*gf='\0';
    gf=strstr(clean,"_GGUF"); if(gf)*gf='\0';
    snprintf(e->full_name, sizeof(e->full_name), "%s", clean);

    make_short_name(clean, e->short_name, sizeof(e->short_name));
    extract_family(clean, e->family, sizeof(e->family));
    extract_quant(best_file, e->quant, sizeof(e->quant));

    e->size_mb = (best_size>0) ? best_size/(1024.0*1024.0) : 0;
    e->ram_needed = (e->size_mb*1.3)/1024.0;
    e->params_b = guess_params(clean);
    e->context_len = guess_context(clean);
    e->has_vision = detect_vision(clean);
    e->has_tools = detect_tools(clean);
    e->has_code = detect_code(clean);
    e->has_voice = detect_voice(clean);
    e->has_stt = detect_stt(clean);
    e->type = classify_model(clean);
    e->min_tier = tier_from_params(e->params_b);

    return 1;
}

/* ── Dedup ────────────────────────────────────────────────────────── */

static int already_have(const model_entry_t* list, int count, const char* full_name) {
    for (int i=0;i<count;i++) {
        if (strcmp(list[i].full_name, full_name)==0) return 1;
        if (strcmp(list[i].filename, full_name)==0) return 1;
    }
    return 0;
}

/* Also check by filename */
static int already_have_file(const model_entry_t* list, int count, const char* filename) {
    for (int i=0;i<count;i++)
        if (strcmp(list[i].filename, filename)==0) return 1;
    return 0;
}

/* ── Fetch catalog for a specific type ────────────────────────────── */

int catalog_fetch(model_entry_t* out, int max, model_type_t type) {
    printf("  Fetching %s models", catalog_type_name(type));
    fflush(stdout);
    int count = 0;
    const char** queries = get_queries(type);

    for (int q=0; queries[q] && count<max; q++) {
        char* json = fetch_json(queries[q], 1024*1024);
        if (!json) continue;
        printf("."); fflush(stdout);

        const char* cursor = json;
        while (count < max) {
            const char* id_pos = strstr(cursor, "\"modelId\"");
            if (!id_pos) id_pos = strstr(cursor, "\"id\"");
            if (!id_pos) break;
            const char* val = strchr(id_pos, ':');
            if (!val) break; val++;
            while (*val==' '||*val=='\t') val++;

            char repo[256]={0};
            if (!json_str(val, repo, sizeof(repo))) { cursor=val+1; continue; }
            cursor = val+strlen(repo)+2;
            if (!strstr(repo,"GGUF")&&!strstr(repo,"gguf")) continue;

            model_entry_t tmp;
            if (!parse_repo(repo, &tmp)) continue;
            if (tmp.params_b < 0.1 && type != MTYPE_VOICE && type != MTYPE_STT) continue;
            if (already_have(out, count, tmp.full_name)) continue;
            if (already_have_file(out, count, tmp.filename)) continue;

            tmp.type = type; /* override with what we searched for */
            out[count++] = tmp;
        }
        free(json);
    }

    printf(" done! (%d)\n\n", count);
    fflush(stdout);

    /* Sort by params ascending */
    for (int i=0;i<count-1;i++)
        for (int j=i+1;j<count;j++)
            if (out[i].params_b > out[j].params_b) {
                model_entry_t t=out[i]; out[i]=out[j]; out[j]=t;
            }
    return count;
}

/* ── Cache ────────────────────────────────────────────────────────── */

void catalog_save_cache(const model_entry_t* entries, int count) {
    char path[1024];
    cli_get_catalog_path(path, sizeof(path));
    cli_ensure_dirs();
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(&count, sizeof(int), 1, f);
    fwrite(entries, sizeof(model_entry_t), count, f);
    fclose(f);
}

int catalog_load_cache(model_entry_t* out, int max) {
    char path[1024];
    cli_get_catalog_path(path, sizeof(path));
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int count=0;
    if (fread(&count,sizeof(int),1,f)!=1) { fclose(f); return 0; }
    if (count<=0||count>max) { fclose(f); return 0; }
    int r=(int)fread(out, sizeof(model_entry_t), count, f);
    fclose(f);
    return r;
}

int catalog_refresh(model_entry_t* out, int max, model_type_t type) {
    int count = catalog_fetch(out, max, type);
    if (count > 0) catalog_save_cache(out, count);
    return count;
}

/* ── Local check ──────────────────────────────────────────────────── */

static int model_is_local(const model_entry_t* m) {
    char path[1024], mdir[1024];
    cli_get_models_dir(mdir, sizeof(mdir));
    snprintf(path, sizeof(path), "%s%c%s", mdir, PATH_SEP, m->filename);
    return cli_file_exists(path);
}

/* ── Parameter filter ─────────────────────────────────────────────── */

typedef struct { const char* label; double min_b; double max_b; } param_filter_t;

static int build_filters(const hw_specs_t* sp, param_filter_t* f, int max) {
    int n=0;
    f[n].label="All sizes"; f[n].min_b=0; f[n].max_b=999; n++;
    f[n].label="Small (up to 1.5B)"; f[n].min_b=0; f[n].max_b=1.5; n++;
    f[n].label="Medium (1.5B - 4B)"; f[n].min_b=1.5; f[n].max_b=4.0; n++;
    if (!sp||!sp->valid||sp->tier>=TIER_MEDIUM) {
        f[n].label="Large (4B - 8B)"; f[n].min_b=4.0; f[n].max_b=8.0; n++;
    }
    if (!sp||!sp->valid||sp->tier>=TIER_HIGH) {
        f[n].label="XL (8B+)"; f[n].min_b=8.0; f[n].max_b=999; n++;
    }
    return n;
}

/* ── Browse with table ────────────────────────────────────────────── */

int catalog_browse(const model_entry_t* entries, int count, const hw_specs_t* specs, model_type_t type, int start_page) {
    double fmin=0, fmax=999;

    if (type == MTYPE_LLM || type == MTYPE_VISION || type == MTYPE_CODE) {
        param_filter_t filters[8];
        int nf = build_filters(specs, filters, 8);

        cli_separator();
        cli_cprint("  SELECT SIZE\n", CLR_CYAN);
        cli_separator();
        printf("\n"); fflush(stdout); cli_delay(100);
        for (int i=0;i<nf;i++) {
            printf("  "); cli_cprint("[",CLR_GREEN); printf("%d",i+1); cli_cprint("]",CLR_GREEN);
            printf(" %s\n", filters[i].label);
            fflush(stdout); cli_delay(60);
        }
        printf("\n  > "); fflush(stdout);
        char buf[16]={0};
        if (!fgets(buf,sizeof(buf),stdin)) return 0;
        int fc=atoi(buf);
        if (fc<1||fc>nf) return 0;
        fmin=filters[fc-1].min_b; fmax=filters[fc-1].max_b;

        /* Clear the size menu, replace with the table */
        cli_clear();
    }

    /* Filter */
    int idx[MAX_CATALOG]; int fcount=0;
    for (int i=0;i<count;i++) {
        if (entries[i].type != type) continue;
        if (entries[i].params_b>=fmin && entries[i].params_b<fmax) idx[fcount++]=i;
        else if (entries[i].params_b<0.1 && (type==MTYPE_VOICE||type==MTYPE_STT)) idx[fcount++]=i;
    }

    if (fcount==0) { printf("\n  No models found in this range.\n\n"); return 0; }

    /* Paginated display — 10 per page */
    int page = start_page;
    int per_page = 10;
    int total_pages = (fcount + per_page - 1) / per_page;
    if (page >= total_pages) page = total_pages - 1;
    if (page < 0) page = 0;

    /* Store filtered indices globally so browse_flow can use them */
    /* (catalog_browse is called for display, browse_flow handles actions) */

    while (1) {
        int start = page * per_page;
        int end = start + per_page;
        if (end > fcount) end = fcount;

        /* Title */
        cli_separator();
        cli_cprint("  ", CLR_CYAN);
        printf("%s MODELS", catalog_type_name(type));
        if (total_pages > 1) printf("  (page %d/%d, %d total)", page+1, total_pages, fcount);
        else printf("  (%d)", fcount);
        printf("\n");
        cli_separator();
        fflush(stdout); cli_delay(150);
        printf("\n");

        /* Header */
        printf("  %-6s%-18s %-7s %-7s %-5s %-7s  %-6s %-6s %-6s %s\n",
               "#", "Name", "Params", "Size", "Ctx", "Quant", "Tools", "Visn", "Code", "Status");
        printf("  %-6s%-18s %-7s %-7s %-5s %-7s  %-6s %-6s %-6s %s\n",
               "----", "------------------", "------", "-------", "-----", "------", "-----", "-----", "-----", "------");
        fflush(stdout); cli_delay(100);

        for (int fi=start; fi<end; fi++) {
            const model_entry_t* m = &entries[idx[fi]];
            int fits = (!specs||!specs->valid) ? 1 : (specs->tier >= m->min_tier);

            char c_num[16], c_name[20], c_params[8], c_size[8], c_ctx[6], c_quant[8];
            char c_tools[4], c_visn[4], c_code[4], c_status[8];

            snprintf(c_num, sizeof(c_num), "[%2d]", fi+1);
            snprintf(c_name, sizeof(c_name), "%.*s", 18, m->short_name);

            if (m->params_b > 0) snprintf(c_params, sizeof(c_params), "%.1fB", m->params_b);
            else snprintf(c_params, sizeof(c_params), "-");

            if (m->size_mb >= 1024) snprintf(c_size, sizeof(c_size), "%.1fGB", m->size_mb/1024.0);
            else if (m->size_mb > 0) snprintf(c_size, sizeof(c_size), "%.0fMB", m->size_mb);
            else snprintf(c_size, sizeof(c_size), "-");

            if (m->context_len >= 131072) snprintf(c_ctx, sizeof(c_ctx), "128K");
            else if (m->context_len >= 32768) snprintf(c_ctx, sizeof(c_ctx), "32K");
            else if (m->context_len >= 8192) snprintf(c_ctx, sizeof(c_ctx), "8K");
            else if (m->context_len >= 1024) snprintf(c_ctx, sizeof(c_ctx), "%dK", m->context_len/1024);
            else snprintf(c_ctx, sizeof(c_ctx), "-");

            snprintf(c_quant, sizeof(c_quant), "%.*s", 7, m->quant);
            snprintf(c_tools, sizeof(c_tools), "%s", m->has_tools ? "Yes" : "No");
            snprintf(c_visn, sizeof(c_visn), "%s", m->has_vision ? "Yes" : "No");
            snprintf(c_code, sizeof(c_code), "%s", m->has_code ? "Yes" : "No");
            snprintf(c_status, sizeof(c_status), "%s", model_is_local(m) ? "[DL]" : "N/A");

            if (fits) cli_color(CLR_GREEN); else cli_color(CLR_YELLOW);
            printf("  %-6s", c_num);
            cli_reset();
            printf("%-18s %-7s %-7s %-5s %-7s  %-6s %-6s %-6s %s\n",
                   c_name, c_params, c_size, c_ctx, c_quant,
                   c_tools, c_visn, c_code, c_status);
            fflush(stdout); cli_delay(50);
        }
        printf("\n");

        /* All actions in one prompt */
        printf("  Download: enter numbers (e.g. 1,3,5)\n");
        printf("  ");
        if (total_pages > 1 && page > 0) { cli_cprint("[P]", CLR_CYAN); printf(" Prev  "); }
        if (total_pages > 1 && page < total_pages - 1) { cli_cprint("[N]", CLR_CYAN); printf(" Next  "); }
        cli_cprint("[L]", CLR_CYAN); printf(" Load more  ");
        cli_cprint("[U]", CLR_CYAN); printf(" Custom URL  ");
        cli_cprint("[M]", CLR_RED); printf(" Main menu");
        printf("\n");

        printf("\n  > ");
        fflush(stdout);
        char nav[256]={0};
        if (!fgets(nav, sizeof(nav), stdin)) return 0;
        nav[strcspn(nav, "\n")] = '\0';

        if (nav[0]=='n'||nav[0]=='N') {
            if (page < total_pages - 1) { page++; cli_clear(); continue; }
            else continue;
        } else if (nav[0]=='p'||nav[0]=='P') {
            if (page > 0) { page--; cli_clear(); continue; }
            else continue;
        } else if (nav[0]=='m'||nav[0]=='M') {
            return 0;
        } else if (nav[0]=='l'||nav[0]=='L') {
            return 'L';
        } else if (nav[0]=='u'||nav[0]=='U') {
            return 'U';
        } else if (nav[0] >= '1' && nav[0] <= '9') {
            /* Download — parse comma-separated numbers */
            char* tok = strtok(nav, ", ");
            while (tok) {
                int choice = atoi(tok);
                if (choice >= 1 && choice <= fcount) {
                    const model_entry_t* dm = &entries[idx[choice-1]];
                    if (!model_is_local(dm)) {
                        /* Inline download — need non-const cast for download_model */
                        download_model(dm);
                    } else {
                        printf("  %s already downloaded.\n", dm->short_name);
                    }
                }
                tok = strtok(NULL, ", ");
            }
            printf("\n  Press Enter to continue...");
            fflush(stdout);
            { char _b[4]; fgets(_b, sizeof(_b), stdin); }
            cli_clear();
            continue;
        } else {
            continue;
        }
    }
    return 0;
}

/* ── Filter helper (for browse_flow to get indices) ───────────────── */

int catalog_filter(const model_entry_t* entries, int count, model_type_t type,
                   double fmin, double fmax, int* idx_out, int max_idx) {
    int n = 0;
    for (int i = 0; i < count && n < max_idx; i++) {
        if (entries[i].type != type) continue;
        if (entries[i].params_b >= fmin && entries[i].params_b < fmax) idx_out[n++] = i;
        else if (entries[i].params_b < 0.1 && (type == MTYPE_VOICE || type == MTYPE_STT)) idx_out[n++] = i;
    }
    return n;
}

/* ── Compat shim ──────────────────────────────────────────────────── */

int catalog_select_and_download(model_entry_t* entries, int count, const hw_specs_t* specs) {
    (void)entries; (void)count; (void)specs;
    return 0;
}
