#include <jni.h>
#include <android/log.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef EDUSOHO_HAS_LLAMA_CPP
#include "llama.h"
#endif

#define LOG_TAG "LlamaEduSohoJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct GgufNativeModelSession {
    std::string model_path;
    int32_t n_ctx = 3072;
    int32_t n_threads = 4;
    uint32_t gguf_version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
    uint64_t file_size_bytes = 0;
    int fd = -1;
    void* mapped_ptr = nullptr;
    size_t mapped_len = 0;

#ifdef EDUSOHO_HAS_LLAMA_CPP
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
#endif
};

std::mutex g_sessions_mutex;
int64_t g_next_handle = 1000L;
std::unordered_map<int64_t, GgufNativeModelSession*> g_sessions;

void closeSessionResources(GgufNativeModelSession* session) {
    if (!session) return;
#ifdef EDUSOHO_HAS_LLAMA_CPP
    if (session->ctx) {
        llama_free(session->ctx);
        session->ctx = nullptr;
    }
    if (session->model) {
        llama_model_free(session->model);
        session->model = nullptr;
    }
#endif
    if (session->mapped_ptr && session->mapped_ptr != MAP_FAILED && session->mapped_len > 0) {
        munmap(session->mapped_ptr, session->mapped_len);
        session->mapped_ptr = nullptr;
        session->mapped_len = 0;
    }
    if (session->fd >= 0) {
        close(session->fd);
        session->fd = -1;
    }
}

std::string extractBetween(const std::string& text, const std::string& startMarker, const std::string& endMarker) {
    size_t s = text.find(startMarker);
    if (s == std::string::npos) return "";
    s += startMarker.size();
    size_t e = text.find(endMarker, s);
    if (e == std::string::npos) return text.substr(s);
    return text.substr(s, e - s);
}

std::string extractCurrentStudentQuestion(const std::string& userBlock) {
    const std::string marker = "Pertanyaan siswa saat ini:";
    size_t pos = userBlock.find(marker);
    if (pos != std::string::npos) {
        std::string sub = userBlock.substr(pos + marker.size());
        size_t stopPos = sub.find("Jawab langsung pertanyaan siswa tersebut");
        if (stopPos != std::string::npos) {
            sub = sub.substr(0, stopPos);
        }
        while (!sub.empty() && (sub.front() == ' ' || sub.front() == '\n' || sub.front() == '\r')) sub.erase(sub.begin());
        while (!sub.empty() && (sub.back() == ' ' || sub.back() == '.' || sub.back() == '\n' || sub.back() == '\r')) sub.pop_back();
        return sub;
    }
    const std::string anchorMarker = "Ingat, jawab langsung tepat pada pokok pertanyaan siswa";
    size_t anchorPos = userBlock.find(anchorMarker);
    if (anchorPos != std::string::npos) {
        return userBlock.substr(0, anchorPos);
    }
    return userBlock;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_spt_edusoho_domain_rpm_gguf_LlamaCppBridge_nativeLoadGgufModel(
        JNIEnv* env,
        jclass /* clazz */,
        jstring modelPath,
        jint nCtx,
        jint nThreads) {
    if (!modelPath) return 0L;
    const char* raw_path = env->GetStringUTFChars(modelPath, nullptr);
    if (!raw_path) return 0L;
    std::string path_str(raw_path);
    env->ReleaseStringUTFChars(modelPath, raw_path);

    int fd = open(path_str.c_str(), O_RDONLY);
    if (fd < 0) {
        LOGE("Gagal membuka berkas GGUF: %s", path_str.c_str());
        return 0L;
    }

    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        LOGE("Ukuran berkas GGUF tidak valid: %s", path_str.c_str());
        close(fd);
        return 0L;
    }

    size_t map_len = static_cast<size_t>(std::min<off_t>(st.st_size, 64LL * 1024LL * 1024LL));
    void* mapped = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED || !mapped) {
        LOGE("mmap gagal pada berkas GGUF: %s", path_str.c_str());
        close(fd);
        return 0L;
    }

    const auto* bytes = static_cast<const uint8_t*>(mapped);
    if (bytes[0] != 0x47 || bytes[1] != 0x47 || bytes[2] != 0x55 || bytes[3] != 0x46) {
        LOGE("Magic bytes bukan GGUF (0x47475546): %s", path_str.c_str());
        munmap(mapped, map_len);
        close(fd);
        return 0L;
    }

    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    std::memcpy(&version, bytes + 4, sizeof(uint32_t));
    std::memcpy(&tensor_count, bytes + 8, sizeof(uint64_t));
    std::memcpy(&kv_count, bytes + 16, sizeof(uint64_t));

    auto* session = new GgufNativeModelSession();
    session->model_path = path_str;
    session->n_ctx = nCtx > 0 ? nCtx : 3072;
    session->n_threads = nThreads > 0 ? nThreads : 4;
    session->gguf_version = version;
    session->tensor_count = tensor_count;
    session->metadata_kv_count = kv_count;
    session->file_size_bytes = static_cast<uint64_t>(st.st_size);
    session->fd = fd;
    session->mapped_ptr = mapped;
    session->mapped_len = map_len;

#ifdef EDUSOHO_HAS_LLAMA_CPP
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    session->model = llama_model_load_from_file(path_str.c_str(), mparams);
    if (!session->model) {
        LOGE("llama_model_load_from_file gagal: %s", path_str.c_str());
        closeSessionResources(session);
        delete session;
        return 0L;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(session->n_ctx);
    cparams.n_batch = 512;
    cparams.n_threads = session->n_threads;
    cparams.n_threads_batch = session->n_threads;
    session->ctx = llama_init_from_model(session->model, cparams);
    if (!session->ctx) {
        LOGE("llama_init_from_model gagal: %s", path_str.c_str());
        closeSessionResources(session);
        delete session;
        return 0L;
    }
#endif

    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    int64_t handle = ++g_next_handle;
    g_sessions[handle] = session;

    LOGI("GGUF berhasil dimuat via JNI: path=%s, version=%u, tensors=%llu, kv=%llu, size=%llu MB",
         path_str.c_str(),
         version,
         static_cast<unsigned long long>(tensor_count),
         static_cast<unsigned long long>(kv_count),
         static_cast<unsigned long long>(session->file_size_bytes / (1024ULL * 1024ULL)));

    return static_cast<jlong>(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_spt_edusoho_domain_rpm_gguf_LlamaCppBridge_nativeCompletionChatMl(
        JNIEnv* env,
        jclass /* clazz */,
        jlong modelHandle,
        jstring chatMlPrompt,
        jint maxTokens,
        jfloat temperature) {
    if (modelHandle == 0L || !chatMlPrompt) {
        return env->NewStringUTF("");
    }

    GgufNativeModelSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        auto it = g_sessions.find(static_cast<int64_t>(modelHandle));
        if (it != g_sessions.end()) {
            session = it->second;
        }
    }
    if (!session || session->file_size_bytes < 10ULL * 1024ULL * 1024ULL) {
        return env->NewStringUTF("");
    }

    const char* raw_prompt = env->GetStringUTFChars(chatMlPrompt, nullptr);
    if (!raw_prompt) return env->NewStringUTF("");
    std::string prompt_str(raw_prompt);
    env->ReleaseStringUTFChars(chatMlPrompt, raw_prompt);

#ifdef EDUSOHO_HAS_LLAMA_CPP
    if (session->model && session->ctx) {
        // 1. Bersihkan KV Cache sesi sebelum mengevaluasi giliran pertanyaan baru
        llama_kv_self_clear(session->ctx);

        const llama_vocab* vocab = llama_model_get_vocab(session->model);
        std::vector<llama_token> tokens(prompt_str.size() + 64);
        int n_tokens = llama_tokenize(
                vocab,
                prompt_str.c_str(),
                static_cast<int32_t>(prompt_str.size()),
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                true,
                true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(
                    vocab,
                    prompt_str.c_str(),
                    static_cast<int32_t>(prompt_str.size()),
                    tokens.data(),
                    static_cast<int32_t>(tokens.size()),
                    true,
                    true);
        }
        if (n_tokens > 0) {
            tokens.resize(n_tokens);
            int limit = maxTokens > 0 ? maxTokens : 420;
            int max_input = session->n_ctx - limit - 32;
            if (n_tokens > max_input && max_input > 256) {
                tokens.erase(tokens.begin(), tokens.end() - max_input);
                n_tokens = static_cast<int>(tokens.size());
            }

            // 2. Evaluasi prompt per chunk 512 token agar tidak melebihi n_batch (512)
            bool decode_ok = true;
            const int chunk_size = 512;
            for (int pos = 0; pos < n_tokens; pos += chunk_size) {
                int n_eval = std::min(chunk_size, n_tokens - pos);
                llama_batch batch = llama_batch_get_one(tokens.data() + pos, n_eval);
                if (llama_decode(session->ctx, batch) != 0) {
                    LOGE("llama_decode gagal pada chunk posisi %d", pos);
                    decode_ok = false;
                    break;
                }
            }

            if (decode_ok) {
                auto sparams = llama_sampler_chain_default_params();
                llama_sampler* smpl = llama_sampler_chain_init(sparams);
                llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.85f, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0.0f ? temperature : 0.40f));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

                std::string generated;
                for (int i = 0; i < limit; ++i) {
                    llama_token new_token_id = llama_sampler_sample(smpl, session->ctx, -1);
                    if (llama_vocab_is_eog(vocab, new_token_id)) {
                        break;
                    }
                    char buf[256];
                    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                    if (n > 0) {
                        generated.append(buf, n);
                        if (generated.find("<|im_end|>") != std::string::npos ||
                            generated.find("<|im_start|>") != std::string::npos ||
                            generated.find("<|endoftext|>") != std::string::npos) {
                            break;
                        }
                    }
                    llama_batch next_batch = llama_batch_get_one(&new_token_id, 1);
                    if (llama_decode(session->ctx, next_batch) != 0) {
                        break;
                    }
                }
                llama_sampler_free(smpl);
                if (!generated.empty()) {
                    return env->NewStringUTF(generated.c_str());
                }
            }
        }
    }
#endif

    // Mode Standalone Fallback jika dikompilasi tanpa folder llama.cpp
    std::string user_block = extractBetween(prompt_str, "<|im_start|>user", "<|im_end|>");
    std::string question = extractCurrentStudentQuestion(user_block);
    std::ostringstream oss;
    oss << "[JNI_GGUF_MMAP_ACTIVE:v" << session->gguf_version
        << ":tensors=" << session->tensor_count
        << ":kv=" << session->metadata_kv_count
        << ":mb=" << (session->file_size_bytes / (1024ULL * 1024ULL))
        << "] " << question;
    std::string out = oss.str();
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_spt_edusoho_domain_rpm_gguf_LlamaCppBridge_nativeFreeGgufModel(
        JNIEnv* /* env */,
        jclass /* clazz */,
        jlong modelHandle) {
    if (modelHandle == 0L) return;
    GgufNativeModelSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        auto it = g_sessions.find(static_cast<int64_t>(modelHandle));
        if (it != g_sessions.end()) {
            session = it->second;
            g_sessions.erase(it);
        }
    }
    if (session) {
        closeSessionResources(session);
        delete session;
    }
}
