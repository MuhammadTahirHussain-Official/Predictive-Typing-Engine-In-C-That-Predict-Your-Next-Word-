#include <stdio.h>
#pragma GCC diagnostic ignored "-Wunused-result"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define CLEAR "cls"
#else
#define CLEAR "clear"
#define Sleep(ms) (void)(ms)
#define Beep(freq, dur) (void)(freq), (void)(dur)
#endif

// ============================================================
// CONSTANTS
// ============================================================
#define MAX_WORD_LEN 50
#define MAX_LINE 1000
#define TOP_K 3
#define MAX_SENTENCE_LENGTH 100
#define INITIAL_CAPACITY 256
#define MODEL_FILE "trained_model.txt"
#define SPELL_MODEL_FILE "spell_model.txt"
#define USERS_FILE "users.dat"
#define MAX_TOPICS 20
#define STYLE_WORDS 5
#define SPELL_TOP_K 5
#define SPELL_MAX_DIST 3
#define MAX_NATURAL_WORDS 30
#define MAX_USERNAME 32
#define MAX_PASSWORD 64
#define MAX_USERS 200
#define ADMIN_SECRET_CODE "123a" /* required to create/login admin */

/* Hash-table sizes — must be power-of-2 for fast masking */
#define VOCAB_HT_SIZE (1 << 14)  /* 16 384 buckets  */
#define BIGRAM_HT_SIZE (1 << 16) /* 65 536 buckets  */
#define TRIGRAM_HT_SIZE (1 << 16)
#define SPELL_HT_SIZE (1 << 16)

/*
 * BOX_W = printable characters inside one box row (between ║ and ║).
 * Every box row must pad/truncate its content to exactly BOX_W chars.
 */
#define BOX_W 58 /* total inner width */

// ANSI Colors
#define RS "\033[0m"
#define GR "\033[1;32m"
#define YL "\033[1;33m"
#define CY "\033[1;36m"
#define PU "\033[1;35m"
#define BL "\033[1;34m"
#define RD "\033[1;31m"
#define WH "\033[1;37m"
#define DM "\033[2m"

// ============================================================
// HASH HELPERS
// ============================================================

static unsigned int djb2(const char *s)
{
    unsigned int h = 5381;
    int c;
    while ((c = (unsigned char)*s++))
        h = ((h << 5) + h) ^ (unsigned int)c;
    return h;
}

// ============================================================
// HASH-TABLE BACKED DATA STRUCTURES
// ============================================================

/* Generic open-addressing hash node for word entries */
typedef struct VocabNode
{
    char key[MAX_WORD_LEN];
    int count;
    struct VocabNode *next; /* chaining */
} VocabNode;

typedef struct NgramNode
{
    char ctx[MAX_WORD_LEN * 3];
    char pred[MAX_WORD_LEN];
    int count;
    struct NgramNode *next;
} NgramNode;

typedef struct SpellNode
{
    char word[MAX_WORD_LEN];
    int freq;
    struct SpellNode *next;
} SpellNode;

/* Flat arrays kept in sync for generation/prediction (filled at save-time
   from the hash tables) — gives O(1) hash lookup AND sequential iteration */

typedef struct
{
    /* Hash table */
    VocabNode **vht; /* vocab hash table [VOCAB_HT_SIZE]    */
    NgramNode **bht; /* bigram hash table                   */
    NgramNode **tht; /* trigram hash table                  */

    /* Flat arrays (rebuilt after training / loading) */
    char (*vocab)[MAX_WORD_LEN];
    int *vcnt;
    int vocab_size;

    char (*bctx)[MAX_WORD_LEN * 3];
    char (*bpred)[MAX_WORD_LEN];
    int *bcnt;
    int bigram_count;

    char (*tctx)[MAX_WORD_LEN * 3];
    char (*tpred)[MAX_WORD_LEN];
    int *tcnt;
    int trigram_count;

    int total_bigrams;
    int total_trigrams;

    /* Topic detection (tiny, kept as flat array) */
    char topics[MAX_TOPICS][MAX_WORD_LEN];
    char keywords[MAX_TOPICS][STYLE_WORDS][MAX_WORD_LEN];
    int topic_sc[MAX_TOPICS];
    int topic_count;
} Predictor;

typedef struct
{
    SpellNode **sht; /* spell hash table [SPELL_HT_SIZE] */
    int dict_size;
    int total_words;
} SpellCorrector;

typedef struct
{
    char word[MAX_WORD_LEN];
    float confidence;
    int count;
} Prediction;
typedef struct
{
    char word[MAX_WORD_LEN];
    int dist, freq;
    float score;
} SpellCandidate;

// ---- Auth ----
#define ROLE_USER 0
#define ROLE_ADMIN 1

typedef struct
{
    char username[MAX_USERNAME];
    char pass_hash[16];
    int role;
    char created[20];
} UserRecord;

typedef struct
{
    UserRecord *users;
    int count, capacity;
} UserDB;
typedef struct
{
    int logged_in, role;
    char username[MAX_USERNAME];
} Session;

// ============================================================
// FORWARD DECLARATIONS
// ============================================================
void boot_sequence(void);
void ai_logs(void);
void loading_bar(const char *);
void type_text(const char *);
void type_text_speed(const char *, int);
void type_colored(const char *, const char *, int);
void alert(const char *);
void thinking_animation(int);
void veryx_thinking(const char *);
void veryx_processing(const char *);
void veryx_success(const char *);
void veryx_error(const char *);
void veryx_info(const char *);
void veryx_banner_mini(void);
void veryx_quip(void);
void veryx_quip_for(int context);
void veryx_exit_animation(void);
void veryx_maybe_remark(int chance);
void scan_animation(const char *);
void glitch_text(const char *);
void neural_pulse(int);
void data_stream(int);

/* box helpers */
void box_top(const char *c);
void box_mid(const char *c);
void box_bot(const char *c);
void box_sep(const char *c);
void box_empty(const char *c);
void box_row(const char *bc, const char *text);
void box_rowc(const char *bc, const char *text);
void box_rowf(const char *bc, const char *fmt, ...);
/* right-aligned variants */
void box_right(const char *bc, const char *text);
void box_rightc(const char *bc, const char *text);
void box_rightf(const char *bc, const char *fmt, ...);

void beep_boot(void);
void beep_success(void);
void beep_error(void);
void beep_thinking(void);
void beep_alert(void);
void beep_train(void);
void beep_generate(void);
void beep_spell(void);
void beep_login(void);

void init_predictor(Predictor *);
void free_predictor(Predictor *);
void train_from_file(Predictor *, const char *);
void train_from_string(Predictor *, const char *);
void rebuild_flat(Predictor *);
void detect_topics(Predictor *);
int predict_next_words(Predictor *, const char *, const char *, Prediction[]);
char *generate_ngram_sentence(Predictor *, const char *, int);
char *generate_style_sentence(Predictor *, const char *, int);
void print_predictions(Prediction[], int);
void print_stats(Predictor *);
void save_model(Predictor *, const char *);
void load_model(Predictor *, const char *);
void auto_save(Predictor *);
void auto_load(Predictor *);
void reset_model(Predictor *);

void init_spell(SpellCorrector *);
void free_spell(SpellCorrector *);
void spell_train_from_file(SpellCorrector *, const char *);
void spell_save(SpellCorrector *, const char *);
void spell_load(SpellCorrector *, const char *);
void spell_auto_save(SpellCorrector *);
void spell_auto_load(SpellCorrector *);
void spell_reset(SpellCorrector *);
void spell_print_stats(SpellCorrector *);
int levenshtein(const char *, const char *);
int spell_correct(SpellCorrector *, const char *, SpellCandidate[]);
void spell_correct_interactive(SpellCorrector *);

unsigned long hash_password(const char *);
void init_userdb(UserDB *);
void free_userdb(UserDB *);
void save_userdb(UserDB *);
void load_userdb(UserDB *);
int userdb_find(UserDB *, const char *);
int auth_signup(UserDB *, const char *, const char *, int);
int auth_login(UserDB *, const char *, const char *, Session *);
void read_password(char *, int);

void clear_screen(void);
void wait_for_enter(void);
void clear_input_buffer(void);
void print_banner(void);
void to_lowercase(char *);
void auth_screen(UserDB *, Session *);
void admin_menu(Predictor *, SpellCorrector *, UserDB *, Session *);
void user_menu(Predictor *, SpellCorrector *, Session *);
void build_sentence(Predictor *);
void auto_generate_sentence(Predictor *);
void style_mimic_demo(Predictor *);
/* Games */
void game_word_scramble(Predictor *);
void game_predict_next(Predictor *);
void game_word_duel(Predictor *, SpellCorrector *);
void game_arcade(Predictor *, SpellCorrector *);
void admin_list_users(UserDB *);
void admin_delete_user(UserDB *, const char *);

// ── Beep sequences ──────────────────────────────────────────
void beep_boot(void)
{
    Beep(400, 80);
    Sleep(30);
    Beep(600, 80);
    Sleep(30);
    Beep(800, 80);
    Sleep(30);
    Beep(1000, 80);
    Sleep(30);
    Beep(1200, 150);
}
void beep_success(void)
{
    Beep(800, 100);
    Sleep(40);
    Beep(1000, 100);
    Sleep(40);
    Beep(1200, 200);
}
void beep_error(void)
{
    Beep(400, 200);
    Sleep(80);
    Beep(300, 200);
    Sleep(80);
    Beep(200, 400);
}
void beep_thinking(void)
{
    Beep(600, 60);
    Sleep(120);
    Beep(650, 60);
    Sleep(120);
    Beep(600, 60);
    Sleep(120);
    Beep(700, 60);
}
void beep_alert(void)
{
    Beep(1000, 150);
    Beep(800, 150);
    Sleep(100);
    Beep(1000, 150);
}
void beep_train(void)
{
    for (int i = 0; i < 3; i++)
    {
        Beep(500 + i * 100, 60);
        Sleep(40);
    }
}
void beep_generate(void)
{
    Beep(700, 50);
    Sleep(30);
    Beep(900, 50);
    Sleep(30);
    Beep(700, 50);
    Sleep(30);
    Beep(1100, 100);
}
void beep_spell(void)
{
    Beep(523, 60);
    Sleep(30);
    Beep(659, 60);
    Sleep(30);
    Beep(784, 60);
    Sleep(30);
    Beep(1047, 120);
}
void beep_login(void)
{
    Beep(880, 80);
    Sleep(40);
    Beep(1100, 80);
    Sleep(40);
    Beep(1320, 160);
}

// ============================================================
// ============================================================
// FORMATTING ENGINE v8 — full-width, slogans, witty comments
// ============================================================
/*
 * TERM_W = printable columns inside ║ … ║
 *          Set to terminal_width - 2.
 *          78 fills a standard 80-column terminal perfectly.
 * INDENT = 0  (boxes fill the full screen width)
 */
#define TERM_W 78
#define INDENT 0

#include <stdarg.h>

/* ── ansi_len: count only visible (printable) columns ───────
 * Skips ESC[…m sequences; counts UTF-8 leading bytes once.  */
static int ansi_len(const char *s)
{
    int n = 0;
    while (*s)
    {
        if ((unsigned char)*s == 0x1b && *(s + 1) == '[')
        {
            s += 2;
            while (*s && *s != 'm')
                s++;
            if (*s)
                s++;
        }
        else
        {
            unsigned char c = (unsigned char)*s;
            if (c < 0x80 || c >= 0xC0)
                n++;
            s++;
        }
    }
    return n;
}

/* Print n spaces */
static void _sp(int n)
{
    while (n-- > 0)
        putchar(' ');
}
static void _ind(void) { _sp(INDENT); }

// ── Box primitives (all exactly TERM_W+2 terminal cols) ─────

void box_top(const char *c)
{
    _ind();
    printf("%s╔", c);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╗%s\n", RS);
}
void box_bot(const char *c)
{
    _ind();
    printf("%s╚", c);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╝%s\n", RS);
}
void box_mid(const char *c)
{
    _ind();
    printf("%s╠", c);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╣%s\n", RS);
}
void box_sep(const char *c)
{
    _ind();
    printf("%s╟", c);
    for (int i = 0; i < TERM_W; i++)
        printf("─");
    printf("╢%s\n", RS);
}
void box_empty(const char *c)
{
    _ind();
    printf("%s║%s", c, RS);
    _sp(TERM_W);
    printf("%s║%s\n", c, RS);
}

/* Core row — text already rendered (ANSI ok), pw=printable width */
static void _row(const char *bc, const char *text, int pw)
{
    int pad = TERM_W - pw;
    if (pad < 0)
        pad = 0;
    _ind();
    printf("%s║%s%s", bc, RS, text);
    _sp(pad);
    printf("%s║%s\n", bc, RS);
}

void box_row(const char *bc, const char *t) { _row(bc, t, (int)strlen(t)); }
void box_rowc(const char *bc, const char *t) { _row(bc, t, ansi_len(t)); }
void box_rowf(const char *bc, const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    box_row(bc, buf);
}

/* Right-justify plain */
void box_right(const char *bc, const char *text)
{
    int tw = (int)strlen(text), lp = TERM_W - tw;
    if (lp < 0)
        lp = 0;
    _ind();
    printf("%s║%s", bc, RS);
    _sp(lp);
    printf("%s%s║%s\n", text, bc, RS);
}
/* Right-justify coloured */
void box_rightc(const char *bc, const char *text)
{
    int tw = ansi_len(text), lp = TERM_W - tw;
    if (lp < 0)
        lp = 0;
    _ind();
    printf("%s║%s", bc, RS);
    _sp(lp);
    printf("%s%s║%s\n", text, bc, RS);
}
void box_rightf(const char *bc, const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    box_right(bc, buf);
}

/* Centre plain */
static void __attribute__((unused))
box_center(const char *bc, const char *text)
{
    int tw = (int)strlen(text), lp = (TERM_W - tw) / 2;
    if (lp < 0)
        lp = 0;
    int rp = TERM_W - tw - lp;
    if (rp < 0)
        rp = 0;
    _ind();
    printf("%s║%s", bc, RS);
    _sp(lp);
    printf("%s", text);
    _sp(rp);
    printf("%s║%s\n", bc, RS);
}
/* Centre coloured */
static void box_centerc(const char *bc, const char *text)
{
    int tw = ansi_len(text), lp = (TERM_W - tw) / 2;
    if (lp < 0)
        lp = 0;
    int rp = TERM_W - tw - lp;
    if (rp < 0)
        rp = 0;
    _ind();
    printf("%s║%s", bc, RS);
    _sp(lp);
    printf("%s", text);
    _sp(rp);
    printf("%s║%s\n", bc, RS);
}
/* Two items on same row: left-aligned + right-aligned */
static void __attribute__((unused)) box_lr(const char *bc,
                                           const char *left_col, const char *left,
                                           const char *right_col, const char *right)
{
    int lw = ansi_len(left), rw = ansi_len(right);
    int gap = TERM_W - lw - rw;
    if (gap < 1)
        gap = 1;
    _ind();
    printf("%s║%s%s%s%s", bc, RS, left_col, left, RS);
    _sp(gap);
    printf("%s%s%s%s║%s\n", right_col, right, RS, bc, RS);
}

// ============================================================
// RANDOM WITTY COMMENT SYSTEM
// ============================================================
/*
 * veryx_quip()  — prints a random one-liner comment to the user.
 * Call it anywhere after a successful action.
 * veryx_quip_for(context) — context-aware version.
 * Context codes:  0=generic  1=training  2=sentence  3=spell
 *                 4=login    5=generate  6=stats      7=exit
 */

static const char *QUIPS_GENERIC[] = {
    "💬  Did you know? VERYX learns faster than your coffee cools.",
    "🤓  Fun fact: every word you type makes VERYX smarter.",
    "⚡  VERYX doesn't sleep. Neither does your data.",
    "🧠  The more you train, the sharper the predictions.",
    "🌟  You're not just typing — you're teaching a neural engine.",
    "🎯  Precision is the product of repetition. Keep going.",
    "🔮  VERYX sees patterns where humans see chaos.",
    "💡  Every sentence you write is a brushstroke on a canvas.",
    "🚀  Great writing always starts with a single word. Go.",
    "🎲  Randomness + patterns = intelligence. Classic VERYX.",
    "📡  Transmitting knowledge at the speed of keystrokes.",
    "🏆  Pro tip: train on diverse text for diverse predictions.",
    "🌀  The more you write, the richer your creative world becomes.",
    "✨  Small words, big patterns. That's the VERYX way.",
    "🔬  Fact: the best writers write first and edit later.",
};
static const char *QUIPS_TRAINING[] = {
    "🚂  Training complete! Your model just levelled up.",
    "📖  More data = wiser predictions. Feed the engine!",
    "⚙️   Hash tables make this 100x faster than the old days.",
    "🏋️   The model is stronger now. Time to test it.",
    "📊  VERYX is counting your words so you don't have to worry.",
    "🌱  Every training run plants seeds of future predictions.",
    "⚡  Data ingested. Neural pathways reinforced. Let's go.",
    "🎓  Your writing just graduated into something extraordinary.",
};
static const char *QUIPS_SENTENCE[] = {
    "✍️   Every sentence begins with a single word. You chose well.",
    "📝  Writer's block? Let VERYX finish that thought.",
    "🎭  Language is art. VERYX is your co-author.",
    "🌊  Words flow like water — just point VERYX downstream.",
    "💬  Good sentences start with good words. You're on track.",
};
static const char *QUIPS_SPELL[] = {
    "🔤  Spelling corrected! One typo closer to perfection.",
    "📚  The dictionary never lies. VERYX doesn't either.",
    "✅  Your words are as sharp as your ideas now.",
    "🧹  Clean spelling, clean thinking. Nice work.",
    "🏅  Spell-checked and approved by the VERYX engine.",
};
static const char *QUIPS_LOGIN[] = {
    "🔑  Welcome back. VERYX missed you. (It's been processing.)",
    "🛡️   Identity confirmed. Neural engines ready.",
    "👤  Another session, another chance to build something great.",
    "🌟  Good to see you. Let's make something intelligent today.",
    "⚡  Logged in. Language model loaded. You may proceed.",
};
static const char *QUIPS_GENERATE[] = {
    "✨  Fresh sentence, served warm and ready to impress!",
    "🎲  Weighted randomness at its finest. Enjoy your sentence.",
    "🔮  VERYX reached into the future of your sentence. Behold!",
    "📡  Sentence assembled from your own training data. Meta, right?",
    "🌊  Language flows. VERYX channelled it just for you.",
};
static const char *QUIPS_STATS[] = {
    "📊  Numbers don't lie. Your model is growing nicely.",
    "📈  More writing = sharper completions. The magic checks out.",
    "🧮  Counting tokens so you don't have to. You're welcome.",
    "🔬  Overview complete. Your vocabulary is genuinely impressive!",
    "🏆  These stats would make a linguist jealous.",
};
static const char *QUIPS_EXIT[] = {
    "👋  See you next session. Your model will be waiting.",
    "🌙  Goodbye! VERYX will keep the weights warm.",
    "🚪  Logging off. All data saved. No knowledge lost.",
    "⭐  Until next time — keep your vocabulary sharp.",
    "🔋  VERYX powering down. Data safely persisted.",
};

static void _print_quip(const char *q)
{
    printf("\n");
    box_top(DM);
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "  %s", q);
        box_rowc(DM, buf);
    }
    box_bot(DM);
    printf("\n");
}

void veryx_quip(void)
{
    int n = (int)(sizeof(QUIPS_GENERIC) / sizeof(QUIPS_GENERIC[0]));
    _print_quip(QUIPS_GENERIC[rand() % n]);
}

void veryx_quip_for(int context)
{
    const char **pool;
    int n;
    switch (context)
    {
    case 1:
        pool = QUIPS_TRAINING;
        n = (int)(sizeof(QUIPS_TRAINING) / sizeof(QUIPS_TRAINING[0]));
        break;
    case 2:
        pool = QUIPS_SENTENCE;
        n = (int)(sizeof(QUIPS_SENTENCE) / sizeof(QUIPS_SENTENCE[0]));
        break;
    case 3:
        pool = QUIPS_SPELL;
        n = (int)(sizeof(QUIPS_SPELL) / sizeof(QUIPS_SPELL[0]));
        break;
    case 4:
        pool = QUIPS_LOGIN;
        n = (int)(sizeof(QUIPS_LOGIN) / sizeof(QUIPS_LOGIN[0]));
        break;
    case 5:
        pool = QUIPS_GENERATE;
        n = (int)(sizeof(QUIPS_GENERATE) / sizeof(QUIPS_GENERATE[0]));
        break;
    case 6:
        pool = QUIPS_STATS;
        n = (int)(sizeof(QUIPS_STATS) / sizeof(QUIPS_STATS[0]));
        break;
    case 7:
        pool = QUIPS_EXIT;
        n = (int)(sizeof(QUIPS_EXIT) / sizeof(QUIPS_EXIT[0]));
        break;
    default:
        pool = QUIPS_GENERIC;
        n = (int)(sizeof(QUIPS_GENERIC) / sizeof(QUIPS_GENERIC[0]));
        break;
    }
    _print_quip(pool[rand() % n]);
}

/* ── PROACTIVE TICKER REMARKS ──────────────────────────────
 * Additional pools for mid-session surprise remarks.
 * veryx_maybe_remark(chance) fires with 1-in-chance probability.
 * Call it at the TOP of every menu loop iteration.            */

static const char *REMARKS_IDLE[] = {
    "💭  " CY "VERYX is watching your patterns..." RS,
    "🌀  " PU "The creative engine inside VERYX never sleeps." RS,
    "🔥  " YL "Every word you type becomes training data for tomorrow." RS,
    "💡  " GR "Tip: the more varied your training text, the smarter VERYX gets." RS,
    "⚡  " CY "Fun fact: VERYX processes text faster than you can read it." RS,
    "🎯  " PU "Precision is the product of repetition. Keep training." RS,
    "🧠  " YL "Your model is growing. Can you feel the intelligence rising?" RS,
    "🌟  " GR "Pro tip: Try training on a novel for rich predictions." RS,
    "🔮  " CY "VERYX just predicted you\'d look at this screen. Spooky." RS,
    "📊  " PU "Your creativity + VERYX's memory = something powerful." RS,
    "🎲  " YL "Randomness weighted by frequency. That\'s not luck — that\'s math." RS,
    "🛡️   " GR "Your data is local. No cloud. No leaks. Pure VERYX." RS,
    "🚀  " CY "VERYX v10.0: built for speed, crafted for creativity." RS,
    "🌊  " PU "Language is a river. VERYX channels the current." RS,
    "📡  " YL "Transmitting intelligence at the speed of keystrokes." RS,
};
static const char *REMARKS_MOTIVATE[] = {
    "💪  " GR "You\'re doing great. The model is getting smarter." RS,
    "🏆  " YL "Champions train daily. So does VERYX." RS,
    "⭐  " CY "Every sentence you build is a test of the model\'s learning." RS,
    "🎓  " PU "You\'re not just using a tool — you\'re teaching a language engine." RS,
    "🌱  " GR "Small inputs. Big intelligence. That\'s how VERYX grows." RS,
    "🔬  " YL "The science of language is in your hands right now." RS,
    "🎯  " CY "Stay consistent. The model rewards repetition with accuracy." RS,
    "💎  " PU "Quality training data is worth more than quantity. Choose wisely." RS,
};
static const char *REMARKS_SARCASTIC[] = {
    "😏  " DM "Still here? The model appreciates the company." RS,
    "🤔  " YL "Thinking about what to type? VERYX already predicted it." RS,
    "👀  " CY "VERYX noticed you paused. Great ideas take a moment to form." RS,
    "😴  " DM "VERYX is wide awake and ready. Are you?" RS,
    "🙃  " PU "Training data won\'t add itself. Just saying." RS,
    "🤖  " YL "The model: \'I could predict your next action, but where\'s the fun?\'" RS,
    "⏰  " CY "Time flies when you're on a creative roll!" RS,
    "🎭  " DM "VERYX is in character. Are you?" RS,
};

/*
 * veryx_maybe_remark(chance)
 *   chance = 1/N probability of firing a remark.
 *   Rotates through IDLE → MOTIVATE → SARCASTIC pools.
 *   Uses a static counter so the same pool isn\'t repeated.
 *
 * For menu loops: call veryx_maybe_remark(4) = 25% each iteration.
 */
static int _remark_pool_idx = 0;

void veryx_maybe_remark(int chance)
{
    if (chance < 1)
        chance = 1;
    if ((rand() % chance) != 0)
        return; /* only fire 1/chance times */

    const char **pool;
    int n;
    int pick = _remark_pool_idx % 3;
    _remark_pool_idx++;

    switch (pick)
    {
    case 0:
        pool = REMARKS_IDLE;
        n = (int)(sizeof(REMARKS_IDLE) / sizeof(REMARKS_IDLE[0]));
        break;
    case 1:
        pool = REMARKS_MOTIVATE;
        n = (int)(sizeof(REMARKS_MOTIVATE) / sizeof(REMARKS_MOTIVATE[0]));
        break;
    default:
        pool = REMARKS_SARCASTIC;
        n = (int)(sizeof(REMARKS_SARCASTIC) / sizeof(REMARKS_SARCASTIC[0]));
        break;
    }

    /* Print as a small inline remark (not a full box — less intrusive) */
    printf("\n");
    _ind();
    printf("%s┌─ VERYX says ──────────────────────────────────────────────────────────────┐%s\n", DM, RS);
    {
        /* Centre the remark text */
        const char *txt = pool[rand() % n];
        int tw = ansi_len(txt);
        int lp = (TERM_W - 2 - tw) / 2;
        if (lp < 0)
            lp = 0;
        int rp = TERM_W - 2 - tw - lp;
        if (rp < 0)
            rp = 0;
        _ind();
        printf("%s│%s", DM, RS);
        _sp(lp);
        printf("%s", txt);
        _sp(rp);
        printf("%s│%s\n", DM, RS);
    }
    _ind();
    printf("%s└───────────────────────────────────────────────────────────────────────────┘%s\n", DM, RS);
    printf("\n");
    fflush(stdout);
}

// ============================================================
// VERYX BANNER — full-width, rotating slogans
// ============================================================

/* The 6 slogan pairs rotate on every print_banner() call.
 * Each pair is:  { main_slogan, sub_slogan }               */
static const char *SLOGANS[][2] = {
    {"⚡  Think It. Type It. VERYX Completes It.  ⚡",
     "🧠 Your thoughts deserve a finishing touch — we deliver it"},
    {"🚀  The Smarter You Type, The Smarter We Get.",
     "💡 Every word you write teaches VERYX something new"},
    {"🔮  Words You Haven't Typed Yet Are Already Waiting.",
     "🌟 VERYX sees patterns in your language before you do"},
    {"🎯  Less Thinking. More Writing. Pure Flow.",
     "✨ VERYX handles the heavy lifting so you stay creative"},
    {"🌊  Language Flows. VERYX Channels It.",
     "🎨 Let your ideas pour out — VERYX shapes them beautifully"},
    {"🏆  The Co-Author You Never Knew You Needed.",
     "🦋 From a single word, VERYX builds entire universes of text"},
};
static int _slogan_idx = 0; /* advances each call */

void veryx_banner_mini(void)
{
    box_top(PU);
    box_centerc(PU, YL "⚡ VERYX" RS "  " CY "NEURAL ENGINE  v10.0" RS);
    box_bot(PU);
}

void print_banner(void)
{
    /* ASCII art rows — each exactly 46 printable chars */
    const char *art[] = {
        " ██╗   ██╗███████╗██████╗ ██╗   ██╗██╗  ██╗ ",
        " ██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝╚██╗██╔╝ ",
        " ██║   ██║█████╗  ██████╔╝ ╚████╔╝  ╚███╔╝  ",
        " ╚██╗ ██╔╝██╔══╝  ██╔══██╗  ╚██╔╝   ██╔██╗  ",
        "  ╚████╔╝ ███████╗██║  ██║   ██║   ██╔╝ ██╗ ",
        "   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ",
    };
    int art_w = 46;
    int lpad = (TERM_W - art_w) / 2;
    int rpad = TERM_W - art_w - lpad;
    if (lpad < 0)
    {
        lpad = 0;
    }
    if (rpad < 0)
    {
        rpad = 0;
    }

    /* Pick current slogan pair */
    int ns = (int)(sizeof(SLOGANS) / sizeof(SLOGANS[0]));
    const char *sl1 = SLOGANS[_slogan_idx % ns][0];
    const char *sl2 = SLOGANS[_slogan_idx % ns][1];
    _slogan_idx++;

    /* ── clear any leftover content before drawing ── */
    (void)system(CLEAR);
    printf("\n");

    /* TOP border */
    _ind();
    printf("%s╔", CY);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╗%s\n", RS);

    /* Top accent: left label + right version — built as one string */
    {
        /* left part printable width = len("  ⚡ VERYX NEURAL ENGINE") = 24 */
        /* right part printable width = len("v10.0  ") = 6 */
        const char *left_txt = "  ⚡ VERYX NEURAL ENGINE";
        const char *right_txt = "v10.0  ";
        int lw = (int)strlen(left_txt);  /* 24 */
        int rw = (int)strlen(right_txt); /* 6  */
        int gap = TERM_W - lw - rw;
        if (gap < 1)
            gap = 1;
        _ind();
        printf("%s║%s" CY "%s" RS, CY, RS, left_txt);
        _sp(gap);
        printf(YL "%s" RS "%s║%s\n", right_txt, CY, RS);
    }

    box_sep(CY);
    box_empty(CY);

    /* ASCII art — yellow on dark background */
    for (int r = 0; r < 6; r++)
    {
        _ind();
        printf("%s║%s", CY, RS);
        _sp(lpad);
        printf("%s%s%s", YL, art[r], RS);
        _sp(rpad);
        printf("%s║%s\n", CY, RS);
    }

    box_empty(CY);
    box_sep(CY);

    /* Primary slogan — centred, purple */
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s%s%s", PU, sl1, RS);
        box_centerc(CY, buf);
    }
    /* Secondary slogan — centred, dim */
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s%s%s", DM, sl2, RS);
        box_centerc(CY, buf);
    }

    /* Creator credit line — centred */
    box_centerc(CY, DM "  ✦  Created by  " WH "Muhammad Tahir Hussain" DM "  ✦  " RS);

    box_empty(CY);

    /* Bottom accent — rotating imaginative ticker */
    {
        /* 8 rotating bottom lines — one per banner call */
        static const char *tickers[] = {
            "  🌌  Where imagination finds its voice",
            "  🔭  Watching your thoughts unfold in real time",
            "  🎨  Painting the perfect sentence, stroke by stroke",
            "  🦋  Your words take flight — VERYX gives them wings",
            "  🌊  Dive in with one word — ride the wave to a story",
            "  🏛️   Built on curiosity. Powered by your creativity.",
            "  🎭  Every great sentence begins with a single spark",
            "  🎵  Your ideas, composed into something unforgettable",
        };
        static const char *right_tickers[] = {
            "Think. Type. Create.  ",
            "Every word opens a door  ",
            "Creativity at your fingertips  ",
            "Language, set free  ",
            "Your voice, amplified  ",
            "Write boldly. Write often.  ",
            "Where ideas become sentences  ",
            "The future speaks your language  ",
        };
        static int _tick = 0;
        int nt = 8;
        const char *left_txt = tickers[_tick % nt];
        const char *right_txt = right_tickers[_tick % nt];
        _tick++;
        int lw = (int)strlen(left_txt);
        int rw = (int)strlen(right_txt);
        int gap = TERM_W - lw - rw;
        if (gap < 1)
            gap = 1;
        _ind();
        printf("%s║%s" PU "%s" RS, CY, RS, left_txt);
        _sp(gap);
        printf(CY "%s" RS "%s║%s\n", right_txt, CY, RS);
    }

    /* BOTTOM border */
    _ind();
    printf("%s╚", CY);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╝%s\n\n", RS);
}

// ── VERYX widgets ────────────────────────────────────────────

void veryx_thinking(const char *lbl)
{
    const char *sp2[] = {"◐", "◓", "◑", "◒"};
    const char *br[] = {"🧠", "⚡", "🔮", "💡"};
    printf("\n");
    box_top(PU);
    box_centerc(PU, YL "⚡ VERYX" RS "  " PU "is preparing your answer..." RS);
    box_mid(PU);
    {
        char buf[200];
        snprintf(buf, sizeof(buf), "  📌 " GR "Task:" RS "  " WH "%.60s" RS, lbl);
        box_rowc(PU, buf);
    }
    box_mid(PU);
    for (int f = 0; f < 12; f++)
    {
        char bar[10] = {0};
        for (int b = 0; b < 8; b++)
            bar[b] = (b < (f + 1) * 2 / 3) ? '#' : '.';
        int dots = (f % 4) + 1;
        char ds[6] = {0};
        for (int d = 0; d < dots; d++)
            ds[d] = '.';
        char row[256];
        snprintf(row, sizeof(row),
                 "  %s%s%s  %sREASONING%s%s   %s[%s]   %s%s%s",
                 CY, sp2[f % 4], RS, GR, RS, ds, BL, bar, YL, br[f % 4], RS);
        box_rowc(PU, row);
        fflush(stdout);
        beep_thinking();
        Sleep(150);
        if (f < 11)
            printf("\033[1A\033[2K");
    }
    box_bot(PU);
    fflush(stdout);
}

void veryx_processing(const char *action)
{
    printf("\n");
    box_top(GR);
    {
        char b[200];
        snprintf(b, sizeof(b), "  ⚡ " GR "VERYX" RS "  " CY "%s" RS, action);
        box_rowc(GR, b);
    }
    box_mid(GR);
    {
        int bw = TERM_W - 8;
        _ind();
        printf("%s║%s    [%s", GR, RS, GR);
        for (int i = 0; i < bw; i++)
            printf("░");
        printf("%s]    %s║%s", RS, GR, RS);
        printf("\r");
        _ind();
        printf("%s║%s    [%s", GR, RS, GR);
        fflush(stdout);
        for (int i = 0; i < bw; i++)
        {
            printf("█");
            fflush(stdout);
            Beep(500 + i * 4, 9);
            Sleep(9);
        }
        printf("%s]    %s", RS, RS);
        _sp(TERM_W - (4 + 1 + bw + 1 + 4));
        printf("%s║%s\n", GR, RS);
    }
    box_bot(GR);
    fflush(stdout);
}

void veryx_success(const char *msg)
{
    printf("\n");
    box_top(GR);
    {
        char b[256];
        snprintf(b, sizeof(b), "  ✅  " GR "SUCCESS:" RS "  " WH "%s" RS, msg);
        box_rowc(GR, b);
    }
    box_bot(GR);
    beep_success();
    fflush(stdout);
}

void veryx_error(const char *msg)
{
    printf("\n");
    box_top(RD);
    {
        char b[256];
        snprintf(b, sizeof(b), "  ❌  " RD "ERROR:" RS "  " WH "%s" RS, msg);
        box_rowc(RD, b);
    }
    box_bot(RD);
    beep_error();
    fflush(stdout);
}

void veryx_info(const char *msg)
{
    printf("\n");
    box_top(CY);
    {
        char b[256];
        snprintf(b, sizeof(b), "  ℹ️   " CY "INFO:" RS "  " WH "%s" RS, msg);
        box_rowc(CY, b);
    }
    box_bot(CY);
    fflush(stdout);
}

void loading_bar(const char *task)
{
    printf("\n");
    box_top(YL);
    {
        char b[200];
        snprintf(b, sizeof(b), "  ⏳  " YL "%s" RS, task);
        box_rowc(YL, b);
    }
    box_mid(YL);
    {
        int bw = TERM_W - 8;
        _ind();
        printf("%s║%s    [%s", YL, RS, GR);
        for (int i = 0; i < bw; i++)
            printf(" ");
        printf("%s]    %s║%s", RS, YL, RS);
        printf("\r");
        _ind();
        printf("%s║%s    [%s", YL, RS, GR);
        fflush(stdout);
        for (int i = 0; i < bw; i++)
        {
            printf("█");
            fflush(stdout);
            Beep(600 + i * 4, 10);
            Sleep(10);
        }
        printf("%s]  " GR "✔ Done" RS, RS);
        _sp(TERM_W - (4 + 1 + bw + 1 + 8));
        printf("%s║%s\n", YL, RS);
    }
    box_bot(YL);
    fflush(stdout);
}

void thinking_animation(int sec)
{
    printf("\n");
    box_top(PU);
    box_centerc(PU, YL "⚡" RS "  " PU "VERYX is waking up..." RS);
    box_mid(PU);
    {
        int total = sec * 3, inner = TERM_W;
        int lp = (inner - total) / 2;
        if (lp < 0)
            lp = 0;
        _ind();
        printf("%s║%s", PU, RS);
        _sp(lp);
        printf("%s", CY);
        for (int i = 0; i < total; i++)
        {
            printf("·");
            fflush(stdout);
            beep_thinking();
            Sleep(270);
        }
        printf("%s", RS);
        _sp(inner - total - lp);
        printf("%s║%s\n", PU, RS);
    }
    box_bot(PU);
}

void alert(const char *m)
{
    printf("\n");
    box_top(YL);
    {
        char b[256];
        snprintf(b, sizeof(b), "  🔔  " YL "%s" RS, m);
        box_rowc(YL, b);
    }
    box_bot(YL);
    beep_alert();
}

void scan_animation(const char *label)
{
    printf("\n");
    box_top(CY);
    {
        char b[200];
        snprintf(b, sizeof(b), "  🔍  " CY "%s" RS, label);
        box_rowc(CY, b);
    }
    box_mid(CY);
    {
        int bw = TERM_W - 10;
        _ind();
        printf("%s║%s     [%s", CY, RS, GR);
        for (int i = 0; i < bw; i++)
            printf(" ");
        printf("%s]     %s║%s", RS, CY, RS);
        printf("\r");
        _ind();
        printf("%s║%s     [%s", CY, RS, GR);
        fflush(stdout);
        for (int i = 0; i < bw; i++)
        {
            printf("▓");
            fflush(stdout);
            Beep(600 + i * 4, 8);
            Sleep(8);
        }
        printf("%s]  " GR "✔ OK" RS, RS);
        _sp(TERM_W - (5 + 1 + bw + 1 + 8));
        printf("%s║%s\n", CY, RS);
    }
    box_bot(CY);
    fflush(stdout);
}

// ── Utility ──────────────────────────────────────────────────

void clear_screen(void) { (void)system(CLEAR); }
void clear_input_buffer(void)
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
        ;
}

void wait_for_enter(void)
{
    printf("\n");
    _ind();
    printf("%s╔", DM);
    for (int i = 0; i < TERM_W; i++)
        printf("─");
    printf("╗%s\n", RS);
    {
        const char *txt = "  Press  ENTER  to continue...";
        int tw = (int)strlen(txt), lp = TERM_W - tw;
        if (lp < 0)
            lp = 0;
        _ind();
        printf("%s║%s%s%s%s", DM, RS, DM, txt, RS);
        _sp(lp);
        printf("%s║%s\n", DM, RS);
    }
    _ind();
    printf("%s╚", DM);
    for (int i = 0; i < TERM_W; i++)
        printf("─");
    printf("╝%s\n", RS);
    fflush(stdout);
    getchar();
}

void to_lowercase(char *s)
{
    for (int i = 0; s[i]; i++)
        s[i] = (char)tolower((unsigned char)s[i]);
}

// ── Boot sequence ─────────────────────────────────────────────

void boot_sequence(void)
{
    printf("\n%s", CY);
    /* Art indented INDENT spaces — with full-width box border */
    _ind();
    printf("╔");
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╗%s\n", RS);

    const char *art[] = {
        " ██╗   ██╗███████╗██████╗ ██╗   ██╗██╗  ██╗ ",
        " ██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝╚██╗██╔╝ ",
        " ██║   ██║█████╗  ██████╔╝ ╚████╔╝  ╚███╔╝  ",
        " ╚██╗ ██╔╝██╔══╝  ██╔══██╗  ╚██╔╝   ██╔██╗  ",
        "  ╚████╔╝ ███████╗██║  ██║   ██║   ██╔╝ ██╗ ",
        "   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ",
    };
    int art_w = 46, lpad = (TERM_W - art_w) / 2, rpad = TERM_W - art_w - lpad;
    if (lpad < 0)
    {
        lpad = 0;
    }
    if (rpad < 0)
    {
        rpad = 0;
    }

    _ind();
    printf("%s║%s", CY, RS);
    _sp(TERM_W);
    printf("%s║%s\n", CY, RS);
    for (int r = 0; r < 6; r++)
    {
        _ind();
        printf("%s║%s", CY, RS);
        _sp(lpad);
        printf("%s%s%s", YL, art[r], RS);
        _sp(rpad);
        printf("%s║%s\n", CY, RS);
    }
    _ind();
    printf("%s║%s", CY, RS);
    _sp(TERM_W);
    printf("%s║%s\n", CY, RS);
    _ind();
    printf("%s╚", CY);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╝%s\n", RS);

    fflush(stdout);
    Sleep(300);
    printf("\n");
    /* Initializing line centred */
    {
        const char *msg = "  ⚡  Initializing VERYX Engine  ";
        int tw = (int)strlen(msg), lp = (TERM_W + 2 - tw) / 2;
        if (lp < 0)
            lp = 0;
        _sp(lp);
        printf("%s%s%s", YL, msg, GR);
    }
    for (int i = 0; i < 6; i++)
    {
        printf(".");
        fflush(stdout);
        beep_boot();
        Sleep(320);
    }
    printf("  %s[ ONLINE ]%s\n\n", WH, RS);
    Sleep(200);
}

void ai_logs(void)
{
    const char *ll[] = {
        "  🔧  Loading neural language modules",
        "  ⚡  Starting the intelligent word engine...",
        "  📖  Mounting spell-correction dictionary",
        "  🔐  Loading user authentication database",
        "  🛡️   Setting up role-based access control",
        "  ✅   VERYX v10.0 is now ONLINE",
    };
    const char *cols[] = {DM, DM, DM, DM, DM, GR};
    int n = (int)(sizeof(ll) / sizeof(ll[0]));

    /* Log box */
    box_top(DM);
    for (int i = 0; i < n; i++)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s%s%s", cols[i], ll[i], RS);
        box_rowc(DM, buf);
        fflush(stdout);
        Sleep(80);
        if (i == n - 1)
            beep_success();
        else
            Beep(400 + i * 50, 35);
    }
    box_bot(DM);
    printf("\n");
}

// ── Type helpers ─────────────────────────────────────────────

void type_text_speed(const char *t, int dm)
{
    for (int i = 0; t[i]; i++)
    {
        printf("%c", t[i]);
        fflush(stdout);
        Sleep(dm);
    }
    printf("\n");
}
void type_text(const char *t) { type_text_speed(t, 22); }
void type_colored(const char *t, const char *c, int dm)
{
    printf("%s", c);
    for (int i = 0; t[i]; i++)
    {
        printf("%c", t[i]);
        fflush(stdout);
        Sleep(dm);
    }
    printf("%s", RS);
    fflush(stdout);
}

void glitch_text(const char *text)
{
    const char *gc = "@#$%&*!?~^";
    int len = (int)strlen(text), gl = (int)strlen(gc);
    int lp = (TERM_W + 2 - (int)strlen(text)) / 2;
    if (lp < 0)
        lp = 0;
    for (int p2 = 0; p2 < 3; p2++)
    {
        printf("\r");
        _sp(lp);
        printf("%s", CY);
        for (int i = 0; i < len; i++)
            printf("%c", text[i] == ' ' ? ' ' : gc[rand() % gl]);
        printf("%s", RS);
        fflush(stdout);
        Sleep(80);
    }
    printf("\r");
    _sp(lp);
    printf("%s%s%s\n", WH, text, RS);
    fflush(stdout);
}

void neural_pulse(int n2)
{
    const char *d[] = {"●○○○○", "○●○○○", "○○●○○", "○○○●○", "○○○○●"};
    int lp = (TERM_W + 2 - 10) / 2;
    if (lp < 0)
        lp = 0;
    for (int i = 0; i < n2; i++)
    {
        printf("\r");
        _sp(lp);
        printf("%s%s%s ", CY, d[i % 5], RS);
        fflush(stdout);
        Sleep(120);
    }
    printf("\r");
    _sp(lp + 12);
    printf("\r");
    fflush(stdout);
}

void data_stream(int lines)
{
    const char *s[] = {
        "0xA3F1  >>  LOADING LANGUAGE MATRIX",
        "0xB72C  >>  MAPPING CONTEXT VECTORS",
        "0xC089  >>  SYNCING PREDICTION ENGINE",
        "0xD4E2  >>  CALIBRATING N-GRAM WEIGHTS",
        "0xE5A7  >>  BUILDING HASH TABLE LATTICE",
        "0xFF00  >>  PRIMING VOCABULARY INDEX",
    };
    int lp = (TERM_W + 2 - 38) / 2;
    if (lp < 0)
        lp = 0;
    for (int i = 0; i < lines; i++)
    {
        _sp(lp);
        printf("%s  %s%s\n", DM, s[i % 6], RS);
        fflush(stdout);
        Sleep(90);
    }
}

// ============================================================
// PASSWORD HASH
// ============================================================
unsigned long hash_password(const char *pw)
{
    unsigned long h = 5381;
    int c;
    while ((c = (unsigned char)*pw++))
        h = ((h << 5) + h) + (unsigned long)c;
    return h;
}

// ============================================================
// USER DATABASE
// ============================================================
void init_userdb(UserDB *db)
{
    db->capacity = 16;
    db->count = 0;
    db->users = (UserRecord *)malloc(db->capacity * sizeof(UserRecord));
    if (!db->users)
    {
        fprintf(stderr, "Fatal userdb\n");
        exit(1);
    }
    memset(db->users, 0, db->capacity * sizeof(UserRecord));
}
void free_userdb(UserDB *db)
{
    if (db->users)
    {
        free(db->users);
        db->users = NULL;
    }
    db->count = 0;
}

void save_userdb(UserDB *db)
{
    FILE *f = fopen(USERS_FILE, "w");
    if (!f)
    {
        veryx_error("Cannot save users file");
        return;
    }
    fprintf(f, "USERS:%d\n", db->count);
    for (int i = 0; i < db->count; i++)
        fprintf(f, "%s|%s|%d|%s\n", db->users[i].username, db->users[i].pass_hash, db->users[i].role, db->users[i].created);
    fclose(f);
}

void load_userdb(UserDB *db)
{
    FILE *f = fopen(USERS_FILE, "r");
    if (!f)
        return;
    char line[256];
    int n = 0;
    if (!fgets(line, sizeof(line), f))
    {
        fclose(f);
        return;
    }
    sscanf(line, "USERS:%d", &n);
    for (int i = 0; i < n; i++)
    {
        if (!fgets(line, sizeof(line), f))
            break;
        line[strcspn(line, "\n")] = 0;
        if (db->count >= db->capacity)
        {
            db->capacity *= 2;
            UserRecord *t = (UserRecord *)realloc(db->users, db->capacity * sizeof(UserRecord));
            if (!t)
                break;
            db->users = t;
        }
        char *p1 = strtok(line, "|"), *p2 = strtok(NULL, "|"), *p3 = strtok(NULL, "|"), *p4 = strtok(NULL, "|");
        if (!p1 || !p2 || !p3)
            continue;
        strncpy(db->users[db->count].username, p1, MAX_USERNAME - 1);
        strncpy(db->users[db->count].pass_hash, p2, 15);
        db->users[db->count].role = atoi(p3);
        if (p4)
            strncpy(db->users[db->count].created, p4, 19);
        db->count++;
    }
    fclose(f);
}

int userdb_find(UserDB *db, const char *u)
{
    for (int i = 0; i < db->count; i++)
        if (!strcmp(db->users[i].username, u))
            return i;
    return -1;
}

int auth_signup(UserDB *db, const char *u, const char *pw, int role)
{
    if (userdb_find(db, u) >= 0)
        return -1;
    if (db->count >= MAX_USERS)
        return -2;
    if (db->count >= db->capacity)
    {
        db->capacity *= 2;
        UserRecord *t = (UserRecord *)realloc(db->users, db->capacity * sizeof(UserRecord));
        if (!t)
            return -2;
        db->users = t;
    }
    int idx = db->count;
    strncpy(db->users[idx].username, u, MAX_USERNAME - 1);
    unsigned long h = hash_password(pw);
    snprintf(db->users[idx].pass_hash, 16, "%010lu", h % 10000000000UL);
    db->users[idx].role = role;
    time_t now = time(NULL);
    struct tm *tm2 = localtime(&now);
    strftime(db->users[idx].created, 20, "%Y-%m-%d %H:%M", tm2);
    db->count++;
    save_userdb(db);
    return 0;
}
int auth_login(UserDB *db, const char *u, const char *pw, Session *s)
{
    int idx = userdb_find(db, u);
    if (idx < 0)
        return -2;
    unsigned long h = hash_password(pw);
    char hs[16];
    snprintf(hs, 16, "%010lu", h % 10000000000UL);
    if (strcmp(hs, db->users[idx].pass_hash))
        return -1;
    s->logged_in = 1;
    s->role = db->users[idx].role;
    strncpy(s->username, u, MAX_USERNAME - 1);
    return 0;
}
void read_password(char *buf, int maxlen)
{
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD m = 0;
    GetConsoleMode(h, &m);
    SetConsoleMode(h, m & ~ENABLE_ECHO_INPUT);
    fgets(buf, maxlen, stdin);
    buf[strcspn(buf, "\n")] = 0;
    SetConsoleMode(h, m);
    printf("\n");
#else
    (void)system("stty -echo 2>/dev/null");
    (void)fgets(buf, maxlen, stdin);
    buf[strcspn(buf, "\n")] = 0;
    (void)system("stty echo 2>/dev/null");
    printf("\n");
#endif
}

// ============================================================
// VERYX EXIT ANIMATION — full-screen animated farewell
// ============================================================

void veryx_exit_animation(void)
{
/* ── helpers ─────────────────────────────────────────── */
/* Print a full-width separator in a given char */
#define FULL_LINE(c, col)                       \
    do                                          \
    {                                           \
        printf("%s", col);                      \
        for (int _i = 0; _i < TERM_W + 2; _i++) \
            printf("%s", c);                    \
        printf("%s\n", RS);                     \
        fflush(stdout);                         \
    } while (0)

/* Print a centred coloured string on a plain background row */
#define CLINE(text, col)                                        \
    do                                                          \
    {                                                           \
        int _tw = ansi_len(text), _lp = (TERM_W + 2 - _tw) / 2; \
        if (_lp < 0)                                            \
            _lp = 0;                                            \
        int _rp = (TERM_W + 2 - _tw - _lp);                     \
        if (_rp < 0)                                            \
            _rp = 0;                                            \
        printf("%s", col);                                      \
        _sp(_lp);                                               \
        printf("%s", text);                                     \
        _sp(_rp);                                               \
        printf("%s\n", RS);                                     \
        fflush(stdout);                                         \
    } while (0)

    (void)system(CLEAR);
    Sleep(120);

    /* ══════════════════════════════════════════════════════
     * PHASE 1 — Particle burst across the screen (top)
     * ══════════════════════════════════════════════════════ */
    const char *sparks[] = {"✦", "★", "◆", "⬟", "◉", "✸", "⊛", "✺", "❋", "✾"};
    const char *spark_cols[] = {YL, GR, CY, PU, WH, RD, YL, CY, GR, PU};
    int ns = 10;
    for (int wave = 0; wave < 3; wave++)
    {
        /* Build one line of random sparks */
        printf("\r");
        for (int col = 0; col < TERM_W + 2; col++)
        {
            if (rand() % 4 == 0)
            {
                int si = rand() % ns;
                printf("%s%s%s", spark_cols[si], sparks[si], RS);
            }
            else
            {
                printf(" ");
            }
        }
        printf("\n");
        fflush(stdout);
        Beep(800 + wave * 200, 80);
        Sleep(180);
    }
    printf("\n");
    Sleep(200);

    /* ══════════════════════════════════════════════════════
     * PHASE 2 — VERYX ASCII title glitch-reveals
     * ══════════════════════════════════════════════════════ */
    const char *art[] = {
        " ██╗   ██╗███████╗██████╗ ██╗   ██╗██╗  ██╗ ",
        " ██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝╚██╗██╔╝ ",
        " ██║   ██║█████╗  ██████╔╝ ╚████╔╝  ╚███╔╝  ",
        " ╚██╗ ██╔╝██╔══╝  ██╔══██╗  ╚██╔╝   ██╔██╗  ",
        "  ╚████╔╝ ███████╗██║  ██║   ██║   ██╔╝ ██╗ ",
        "   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ",
    };
    int art_w = 46;
    int art_lpad = (TERM_W + 2 - art_w) / 2;
    if (art_lpad < 0)
        art_lpad = 0;

    /* Glitch effect: 3 passes of random chars, then reveal line by line */
    const char *glitch_chars = "@#$%&*!?~^><+=|";
    int gc_len = (int)strlen(glitch_chars);
    for (int pass = 0; pass < 4; pass++)
    {
        printf("\033[H"); /* move cursor to top-left */
        /* re-print spark lines as blank */
        for (int r = 0; r < 4; r++)
            printf("%*s\n", TERM_W + 2, "");
        /* glitch art */
        for (int r = 0; r < 6; r++)
        {
            _sp(art_lpad);
            printf("%s", pass < 3 ? CY : YL);
            int al = (int)strlen(art[r]);
            for (int c2 = 0; c2 < al; c2++)
            {
                if (art[r][c2] != ' ' && pass < 3 && rand() % 3 != 0)
                    printf("%c", glitch_chars[rand() % gc_len]);
                else
                    printf("%c", art[r][c2]);
            }
            printf("%s\n", RS);
        }
        fflush(stdout);
        Beep(400 + pass * 150, 60);
        Sleep(120);
    }
    Sleep(300);

    /* ══════════════════════════════════════════════════════
     * PHASE 3 — Animated farewell box builds line by line
     * ══════════════════════════════════════════════════════ */
    (void)system(CLEAR);
    Sleep(80);

    /* Draw top border letter by letter */
    printf("%s║", CY);
    for (int i = 0; i < TERM_W; i++)
    {
        printf("═");
        fflush(stdout);
        Beep(500 + i * 3, 8);
        Sleep(4);
    }
    printf("║%s\n", RS);

    Sleep(100);

    /* Empty row */
    printf("%s║%s", CY, RS);
    _sp(TERM_W);
    printf("%s║%s\n", CY, RS);

    /* Reveal art rows with typewriter */
    for (int r = 0; r < 6; r++)
    {
        int lpad2 = (TERM_W - art_w) / 2;
        if (lpad2 < 0)
            lpad2 = 0;
        int rpad2 = TERM_W - art_w - lpad2;
        if (rpad2 < 0)
            rpad2 = 0;
        printf("%s║%s", CY, RS);
        _sp(lpad2);
        printf("%s", YL);
        int al = (int)strlen(art[r]);
        for (int c2 = 0; c2 < al; c2++)
        {
            printf("%c", art[r][c2]);
            fflush(stdout);
            Sleep(6);
        }
        printf("%s", RS);
        _sp(rpad2);
        printf("%s║%s\n", CY, RS);
        Beep(700 + r * 80, 40);
        Sleep(60);
    }

    /* Separator */
    printf("%s║%s", CY, RS);
    _sp(TERM_W);
    printf("%s║%s\n", CY, RS);
    printf("%s╠", PU);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╣%s\n", RS);
    Sleep(200);

    /* ── Animated farewell text lines ── */
    /* Each line types itself character by character */
    struct
    {
        const char *text;
        const char *col;
        int delay_ms;
    } farewell[] = {
        {"  ✨  Thank you for using VERYX!", GR, 22},
        {"  💡  Every word you wrote made it smarter.", YL, 20},
        {"  🌊  Your ideas are safe — VERYX remembers.", CY, 20},
        {"  🎯  Come back anytime. The engine awaits.", PU, 20},
        {"", WH, 0},
        {"  ✦  Created by  Muhammad Tahir Hussain  ✦", WH, 25},
        {"", WH, 0},
        {"  🚀  VERYX v10.0  —  Until next time...", DM, 18},
    };
    int nf = (int)(sizeof(farewell) / sizeof(farewell[0]));

    for (int i = 0; i < nf; i++)
    {
        const char *txt = farewell[i].text;
        const char *col = farewell[i].col;
        int dl = farewell[i].delay_ms;
        int tlen = (int)strlen(txt);
        int pad = TERM_W - tlen;
        if (pad < 0)
            pad = 0;

        printf("%s║%s", PU, RS);
        printf("%s", col);
        for (int c2 = 0; txt[c2]; c2++)
        {
            printf("%c", txt[c2]);
            fflush(stdout);
            if (dl > 0)
            {
                Beep(600 + c2 * 4, 8);
                Sleep(dl);
            }
        }
        printf("%s", RS);
        _sp(pad);
        printf("%s║%s\n", PU, RS);
        Sleep(120);
    }

    /* Separator before progress bar */
    printf("%s╠", PU);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╣%s\n", RS);
    Sleep(150);

    /* ── SHUTDOWN progress bar ── */
    {
        const char *bar_label = "  🔋  Saving your session and powering down...";
        int tlen = (int)strlen(bar_label), pad = TERM_W - tlen;
        if (pad < 0)
            pad = 0;
        printf("%s║%s", PU, RS);
        printf("%s", DM);
        for (int c2 = 0; bar_label[c2]; c2++)
        {
            printf("%c", bar_label[c2]);
            fflush(stdout);
            Beep(500 + c2 * 3, 9);
            Sleep(16);
        }
        printf("%s", RS);
        _sp(pad);
        printf("%s║%s\n", PU, RS);
        Sleep(100);

        int bw = TERM_W - 8;
        printf("%s║%s    [%s", PU, RS, DM);
        for (int i = 0; i < bw; i++)
            printf("░");
        printf("%s]    %s%s║%s", RS, RS, PU, RS);
        printf("\r%s║%s    [%s", PU, RS, GR);
        fflush(stdout);
        for (int i = 0; i < bw; i++)
        {
            printf("█");
            fflush(stdout);
            Beep(400 + i * 4, 10);
            Sleep(10);
        }
        printf("%s]  " GR "✔" RS, RS);
        _sp(TERM_W - (4 + 1 + bw + 1 + 4));
        printf("%s║%s\n", PU, RS);
    }

    /* Bottom border */
    printf("%s╠", PU);
    for (int i = 0; i < TERM_W; i++)
        printf("═");
    printf("╣%s\n", RS);
    Sleep(120);

    /* ══════════════════════════════════════════════════════
     * PHASE 4 — Final spark shower (bottom)
     * ══════════════════════════════════════════════════════ */

    /* Colour-wave rows */
    const char *wave_cols[] = {RD, YL, GR, CY, PU, BL, CY, GR, YL};
    int nwc = 9;
    for (int w = 0; w < TERM_W + 2; w += 2)
    {
        /* Print a single-char wide advancing column */
        printf("\033[%d;%dH", 22, w + 1); /* move cursor */
        printf("%s%s%s", wave_cols[w % nwc], "▓", RS);
        fflush(stdout);
        Beep(300 + w * 5, 10);
        Sleep(6);
    }
    printf("\n\n");

    /* One last centred goodbye */
    {
        const char *bye = "  ⚡  VERYX signing off. See you on the other side.  ⚡  ";
        int blen = (int)strlen(bye), lp = (TERM_W + 2 - blen) / 2;
        if (lp < 0)
            lp = 0;
        _sp(lp);
        printf("%s", PU);
        for (int c2 = 0; bye[c2]; c2++)
        {
            printf("%c", bye[c2]);
            fflush(stdout);
            Beep(800 + c2 * 3, 10);
            Sleep(22);
        }
        printf("%s\n\n", RS);
    }

    /* ── Final musical goodbye beep sequence ── */
    int melody[] = {523, 587, 659, 698, 784, 880, 988, 1047};
    int mlen = 8;
    for (int i = 0; i < mlen; i++)
    {
        Beep(melody[i], 120);
        Sleep(50);
    }
    Sleep(600);

    /* ── Fade out: dim the screen with overwrite rows ── */
    for (int row = 0; row < 6; row++)
    {
        printf("%s", DM);
        for (int c2 = 0; c2 < TERM_W + 2; c2++)
            printf("░");
        printf("%s\n", RS);
        fflush(stdout);
        Sleep(80);
    }
    printf("\n");
    Sleep(400);

    (void)system(CLEAR);

#undef FULL_LINE
#undef CLINE
}

// ============================================================
// AUTH SCREEN
// ============================================================
void auth_screen(UserDB *db, Session *sess)
{
    sess->logged_in = 0;
    while (!sess->logged_in)
    {
        print_banner();
        printf("\n");
        box_top(CY);
        box_centerc(CY, YL "⚡ VERYX" RS "  " CY "ACCESS PORTAL" RS);
        box_sep(CY);
        box_rowc(CY, "  " GR "1." RS "  🔑  Login  as " WH "User" RS);
        box_rowc(CY, "  " GR "2." RS "  🛡️   Login  as " YL "Admin" RS "  " DM "(requires admin code)" RS);
        box_sep(CY);
        box_rowc(CY, "  " GR "3." RS "  📝  Sign Up as " WH "User" RS);
        box_rowc(CY, "  " GR "4." RS "  📝  Sign Up as " YL "Admin" RS "  " DM "(requires admin code)" RS);
        box_sep(CY);
        box_rowc(CY, "  " RD "5." RS "  🚪  Exit VERYX");
        box_bot(CY);
        printf("\n  %s▶ Choice:%s ", PU, RS);
        fflush(stdout);
        int ch;
        if (scanf("%d", &ch) != 1)
        {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();
        if (ch == 5)
        {
            veryx_exit_animation();
            exit(0);
        }
        if (ch < 1 || ch > 4)
            continue;

        int is_signup = (ch == 3 || ch == 4);
        int target_role = (ch == 2 || ch == 4) ? ROLE_ADMIN : ROLE_USER;
        const char *rl = (target_role == ROLE_ADMIN) ? "Admin" : "User";

        /* ── Admin code gate ─────────────────────────────────── */
        if (target_role == ROLE_ADMIN)
        {
            char acode[32] = "";
            printf("\n");
            box_top(YL);
            box_centerc(YL, YL "🔐  ADMIN VERIFICATION REQUIRED" RS);
            box_mid(YL);
            box_rowc(YL, "  " DM "Enter the admin secret code to proceed." RS);
            box_bot(YL);
            printf("  %s🔐 Admin Code:%s ", YL, RS);
            fflush(stdout);
            read_password(acode, sizeof(acode));
            if (strcmp(acode, ADMIN_SECRET_CODE) != 0)
            {
                printf("\n");
                glitch_text("  ██ ADMIN ACCESS DENIED ██");
                beep_error();
                Sleep(500);
                box_top(RD);
                box_rowc(RD, "  " RD "❌  Invalid admin code — access blocked." RS);
                box_bot(RD);
                Sleep(1600);
                continue;
            }
            box_top(GR);
            box_rowc(GR, "  " GR "✅  Admin code verified. Proceeding..." RS);
            box_bot(GR);
            beep_success();
            Sleep(600);
        }

        print_banner();
        printf("\n");
        if (is_signup)
        {
            box_top(GR);
            box_centerc(GR, GR "📝  SIGN UP  ─  " WH RS);
            {
                char b[128];
                snprintf(b, sizeof(b), "  Creating %s account...", rl);
                box_row(GR, b);
            }
            box_bot(GR);
        }
        else
        {
            box_top(PU);
            box_centerc(PU, PU "🔐  LOGIN  ─  " WH RS);
            {
                char b[128];
                snprintf(b, sizeof(b), "  Authenticating as %s", rl);
                box_row(PU, b);
            }
            box_bot(PU);
        }

        char username[MAX_USERNAME] = "", password[MAX_PASSWORD] = "";
        printf("\n  %s🔑 %sUsername%s : ", CY, WH, RS);
        fflush(stdout);
        if (fgets(username, sizeof(username), stdin) == NULL)
            continue;
        username[strcspn(username, "\n")] = 0;
        if ((int)strlen(username) < 2)
        {
            veryx_error("Username must be at least 2 characters");
            Sleep(1200);
            continue;
        }
        printf("  %s🔒 %sPassword%s : %s", CY, WH, RS, DM);
        fflush(stdout);
        read_password(password, sizeof(password));
        printf("%s", RS);
        if ((int)strlen(password) < 4)
        {
            veryx_error("Password must be at least 4 characters");
            Sleep(1200);
            continue;
        }

        if (is_signup)
        {
            char confirm[MAX_PASSWORD] = "";
            printf("  %s🔒 %sConfirm%s  : %s", CY, WH, RS, DM);
            fflush(stdout);
            read_password(confirm, sizeof(confirm));
            printf("%s", RS);
            if (strcmp(password, confirm))
            {
                veryx_error("Passwords do not match");
                Sleep(1200);
                continue;
            }
            printf("\n  %s⚡ VERYX%s  creating account", YL, RS);
            for (int i = 0; i < 4; i++)
            {
                printf(".");
                fflush(stdout);
                Sleep(220);
            }
            int r = auth_signup(db, username, password, target_role);
            if (r == -1)
            {
                veryx_error("Username already taken");
                Sleep(1400);
                continue;
            }
            if (r == -2)
            {
                veryx_error("Server full");
                Sleep(1400);
                continue;
            }
            auth_login(db, username, password, sess);
            printf("\n");
            box_top(GR);
            box_centerc(GR, GR "🎉  Account Created Successfully!" RS);
            box_mid(GR);
            {
                char b[256];
                snprintf(b, sizeof(b), "  👤 " GR "%s" RS "  ·  Role: " YL "%s" RS "  ·  Welcome aboard!", username, rl);
                box_rowc(GR, b);
            }
            box_bot(GR);
            beep_login();
            Sleep(1400);
        }
        else
        {
            int idx = userdb_find(db, username);
            if (idx >= 0 && db->users[idx].role != target_role)
            {
                veryx_error("Wrong portal — this account belongs to a different role");
                Sleep(1600);
                continue;
            }
            printf("\n  %s⚡ VERYX%s  authenticating", YL, RS);
            for (int i = 0; i < 4; i++)
            {
                printf(".");
                fflush(stdout);
                Sleep(220);
            }
            int r = auth_login(db, username, password, sess);
            if (r == -2)
            {
                veryx_error("Username not found — please sign up first");
                Sleep(1400);
                continue;
            }
            if (r == -1)
            {
                printf("\n");
                glitch_text("  ██ ACCESS DENIED ██");
                beep_error();
                Sleep(800);
                veryx_error("Incorrect password — try again");
                Sleep(1200);
                continue;
            }
            printf("\n");
            box_top(GR);
            box_centerc(GR, GR "✅  Authentication Successful" RS);
            box_mid(GR);
            {
                char b[256];
                const char *role_col = (target_role == ROLE_ADMIN) ? YL : WH;
                snprintf(b, sizeof(b), "  👤 " GR "%s" RS "  ·  Role: %s%s" RS "  ·  Session active", username, role_col, rl);
                box_rowc(GR, b);
            }
            box_bot(GR);
            beep_login();
            veryx_quip_for(4);
            Sleep(1200);
        }
    }
}

// ============================================================
//  HASH-TABLE PREDICTOR — O(1) insert & lookup
// ============================================================

void init_predictor(Predictor *p)
{
    p->vht = (VocabNode **)calloc(VOCAB_HT_SIZE, sizeof(VocabNode *));
    p->bht = (NgramNode **)calloc(BIGRAM_HT_SIZE, sizeof(NgramNode *));
    p->tht = (NgramNode **)calloc(TRIGRAM_HT_SIZE, sizeof(NgramNode *));
    if (!p->vht || !p->bht || !p->tht)
    {
        veryx_error("Predictor alloc failed");
        exit(1);
    }
    p->vocab = NULL;
    p->vcnt = NULL;
    p->vocab_size = 0;
    p->bctx = NULL;
    p->bpred = NULL;
    p->bcnt = NULL;
    p->bigram_count = 0;
    p->tctx = NULL;
    p->tpred = NULL;
    p->tcnt = NULL;
    p->trigram_count = 0;
    p->total_bigrams = p->total_trigrams = 0;
    p->topic_count = 0;
}

void free_predictor(Predictor *p)
{
    /* free hash chains */
    if (p->vht)
    {
        for (int i = 0; i < VOCAB_HT_SIZE; i++)
        {
            VocabNode *n = p->vht[i];
            while (n)
            {
                VocabNode *nx = n->next;
                free(n);
                n = nx;
            }
        }
        free(p->vht);
        p->vht = NULL;
    }
    if (p->bht)
    {
        for (int i = 0; i < BIGRAM_HT_SIZE; i++)
        {
            NgramNode *n = p->bht[i];
            while (n)
            {
                NgramNode *nx = n->next;
                free(n);
                n = nx;
            }
        }
        free(p->bht);
        p->bht = NULL;
    }
    if (p->tht)
    {
        for (int i = 0; i < TRIGRAM_HT_SIZE; i++)
        {
            NgramNode *n = p->tht[i];
            while (n)
            {
                NgramNode *nx = n->next;
                free(n);
                n = nx;
            }
        }
        free(p->tht);
        p->tht = NULL;
    }
    /* free flat arrays */
    free(p->vocab);
    free(p->vcnt);
    p->vocab = NULL;
    p->vcnt = NULL;
    free(p->bctx);
    free(p->bpred);
    free(p->bcnt);
    p->bctx = NULL;
    p->bpred = NULL;
    p->bcnt = NULL;
    free(p->tctx);
    free(p->tpred);
    free(p->tcnt);
    p->tctx = NULL;
    p->tpred = NULL;
    p->tcnt = NULL;
    p->vocab_size = p->bigram_count = p->trigram_count = 0;
    p->total_bigrams = p->total_trigrams = 0;
    p->topic_count = 0;
}

/* O(1) vocab insert */
static void vocab_add(Predictor *p, const char *w)
{
    unsigned int h = djb2(w) & (VOCAB_HT_SIZE - 1);
    for (VocabNode *n = p->vht[h]; n; n = n->next)
        if (!strcmp(n->key, w))
        {
            n->count++;
            return;
        }
    VocabNode *nd = (VocabNode *)malloc(sizeof(VocabNode));
    if (!nd)
        return;
    strncpy(nd->key, w, MAX_WORD_LEN - 1);
    nd->key[MAX_WORD_LEN - 1] = 0;
    nd->count = 1;
    nd->next = p->vht[h];
    p->vht[h] = nd;
}

/* O(1) bigram insert — hash by ctx only so wb() and predict_next_words() find it */
static void bigram_add(Predictor *p, const char *ctx, const char *pred)
{
    p->total_bigrams++;
    unsigned int h = djb2(ctx) & (BIGRAM_HT_SIZE - 1);
    for (NgramNode *n = p->bht[h]; n; n = n->next)
        if (!strcmp(n->ctx, ctx) && !strcmp(n->pred, pred))
        {
            n->count++;
            return;
        }
    NgramNode *nd = (NgramNode *)malloc(sizeof(NgramNode));
    if (!nd)
        return;
    strncpy(nd->ctx, ctx, MAX_WORD_LEN * 3 - 1);
    nd->ctx[MAX_WORD_LEN * 3 - 1] = 0;
    strncpy(nd->pred, pred, MAX_WORD_LEN - 1);
    nd->pred[MAX_WORD_LEN - 1] = 0;
    nd->count = 1;
    nd->next = p->bht[h];
    p->bht[h] = nd;
}

/* O(1) trigram insert — hash by ctx only so wt() and predict_next_words() find it */
static void trigram_add(Predictor *p, const char *ctx, const char *pred)
{
    p->total_trigrams++;
    unsigned int h = djb2(ctx) & (TRIGRAM_HT_SIZE - 1);
    for (NgramNode *n = p->tht[h]; n; n = n->next)
        if (!strcmp(n->ctx, ctx) && !strcmp(n->pred, pred))
        {
            n->count++;
            return;
        }
    NgramNode *nd = (NgramNode *)malloc(sizeof(NgramNode));
    if (!nd)
        return;
    strncpy(nd->ctx, ctx, MAX_WORD_LEN * 3 - 1);
    nd->ctx[MAX_WORD_LEN * 3 - 1] = 0;
    strncpy(nd->pred, pred, MAX_WORD_LEN - 1);
    nd->pred[MAX_WORD_LEN - 1] = 0;
    nd->count = 1;
    nd->next = p->tht[h];
    p->tht[h] = nd;
}

/* Rebuild flat arrays from hash tables (called after training / loading) */
void rebuild_flat(Predictor *p)
{
    free(p->vocab);
    free(p->vcnt);
    free(p->bctx);
    free(p->bpred);
    free(p->bcnt);
    free(p->tctx);
    free(p->tpred);
    free(p->tcnt);

    /* ── VOCAB: single pass, realloc doubling ── */
    int vc = 0, vcap = 256;
    p->vocab = (char (*)[MAX_WORD_LEN])malloc((size_t)vcap * MAX_WORD_LEN);
    p->vcnt = (int *)malloc((size_t)vcap * sizeof(int));
    for (int i = 0; i < VOCAB_HT_SIZE; i++)
        for (VocabNode *n = p->vht[i]; n; n = n->next)
        {
            if (vc == vcap)
            {
                vcap *= 2;
                p->vocab = (char (*)[MAX_WORD_LEN])realloc(p->vocab, (size_t)vcap * MAX_WORD_LEN);
                p->vcnt = (int *)realloc(p->vcnt, (size_t)vcap * sizeof(int));
            }
            memcpy(p->vocab[vc], n->key, MAX_WORD_LEN);
            p->vocab[vc][MAX_WORD_LEN - 1] = 0;
            p->vcnt[vc] = n->count;
            vc++;
        }
    p->vocab_size = vc;

    /* ── BIGRAMS: single pass, realloc doubling ── */
    int bc = 0, bcap = 1024;
    p->bctx = (char (*)[MAX_WORD_LEN * 3]) malloc((size_t)bcap * MAX_WORD_LEN * 3);
    p->bpred = (char (*)[MAX_WORD_LEN])malloc((size_t)bcap * MAX_WORD_LEN);
    p->bcnt = (int *)malloc((size_t)bcap * sizeof(int));
    for (int i = 0; i < BIGRAM_HT_SIZE; i++)
        for (NgramNode *n = p->bht[i]; n; n = n->next)
        {
            if (bc == bcap)
            {
                bcap *= 2;
                p->bctx = (char (*)[MAX_WORD_LEN * 3]) realloc(p->bctx, (size_t)bcap * MAX_WORD_LEN * 3);
                p->bpred = (char (*)[MAX_WORD_LEN])realloc(p->bpred, (size_t)bcap * MAX_WORD_LEN);
                p->bcnt = (int *)realloc(p->bcnt, (size_t)bcap * sizeof(int));
            }
            memcpy(p->bctx[bc], n->ctx, MAX_WORD_LEN * 3);
            p->bctx[bc][MAX_WORD_LEN * 3 - 1] = 0;
            memcpy(p->bpred[bc], n->pred, MAX_WORD_LEN);
            p->bpred[bc][MAX_WORD_LEN - 1] = 0;
            p->bcnt[bc] = n->count;
            bc++;
        }
    p->bigram_count = bc;

    /* ── TRIGRAMS: single pass, realloc doubling ── */
    int tc = 0, tcap = 1024;
    p->tctx = (char (*)[MAX_WORD_LEN * 3]) malloc((size_t)tcap * MAX_WORD_LEN * 3);
    p->tpred = (char (*)[MAX_WORD_LEN])malloc((size_t)tcap * MAX_WORD_LEN);
    p->tcnt = (int *)malloc((size_t)tcap * sizeof(int));
    for (int i = 0; i < TRIGRAM_HT_SIZE; i++)
        for (NgramNode *n = p->tht[i]; n; n = n->next)
        {
            if (tc == tcap)
            {
                tcap *= 2;
                p->tctx = (char (*)[MAX_WORD_LEN * 3]) realloc(p->tctx, (size_t)tcap * MAX_WORD_LEN * 3);
                p->tpred = (char (*)[MAX_WORD_LEN])realloc(p->tpred, (size_t)tcap * MAX_WORD_LEN);
                p->tcnt = (int *)realloc(p->tcnt, (size_t)tcap * sizeof(int));
            }
            memcpy(p->tctx[tc], n->ctx, MAX_WORD_LEN * 3);
            p->tctx[tc][MAX_WORD_LEN * 3 - 1] = 0;
            memcpy(p->tpred[tc], n->pred, MAX_WORD_LEN);
            p->tpred[tc][MAX_WORD_LEN - 1] = 0;
            p->tcnt[tc] = n->count;
            tc++;
        }
    p->trigram_count = tc;
}

void train_from_string(Predictor *p, const char *text)
{
    size_t tl = strlen(text);
    char *tc = (char *)malloc(tl + 1);
    if (!tc)
        return;
    memcpy(tc, text, tl + 1);
    char *ws[MAX_LINE];
    int wc = 0;
    char *tok = strtok(tc, " .,!?;:\n\t");
    while (tok && wc < MAX_LINE - 1)
    {
        to_lowercase(tok);
        vocab_add(p, tok);
        ws[wc++] = tok;
        tok = strtok(NULL, " .,!?;:\n\t");
    }
    for (int i = 0; i < wc; i++)
    {
        if (i < wc - 1)
            bigram_add(p, ws[i], ws[i + 1]);
        if (i < wc - 2)
        {
            char ctx[MAX_WORD_LEN * 3];
            snprintf(ctx, sizeof(ctx), "%s %s", ws[i], ws[i + 1]);
            trigram_add(p, ctx, ws[i + 2]);
        }
    }
    free(tc);
}

static const char *TOPIC_SEEDS[MAX_TOPICS] = {"programming", "python", "machine", "learning", "artificial", "data", "science", "neural", "network", "deep", "weather", "love", "like", "enjoy", "cat", "dog", "animal", "fox", "fun", "code"};

void detect_topics(Predictor *p)
{
    p->topic_count = 0;
    if (!p->vocab_size || !p->bigram_count)
        return;
    for (int k = 0; k < MAX_TOPICS && p->topic_count < MAX_TOPICS; k++)
    {
        const char *seed = TOPIC_SEEDS[k];
        /* check vocab hash */
        unsigned int vh = djb2(seed) & (VOCAB_HT_SIZE - 1);
        int found = 0;
        for (VocabNode *n = p->vht[vh]; n; n = n->next)
            if (!strcmp(n->key, seed))
            {
                found = 1;
                break;
            }
        if (!found)
            continue;
        int already = 0;
        for (int t = 0; t < p->topic_count; t++)
            if (!strcmp(p->topics[t], seed))
            {
                already = 1;
                break;
            }
        if (already)
            continue;
        strncpy(p->topics[p->topic_count], seed, MAX_WORD_LEN - 1);
        p->topics[p->topic_count][MAX_WORD_LEN - 1] = 0;
        p->topic_sc[p->topic_count] = 0;
        for (VocabNode *n = p->vht[vh]; n; n = n->next)
            if (!strcmp(n->key, seed))
            {
                p->topic_sc[p->topic_count] = n->count;
                break;
            }
        /* collect top bigram followers */
        char fw[64][MAX_WORD_LEN];
        int fc[64], fn = 0;
        for (int i = 0; i < p->bigram_count && fn < 64; i++)
            if (!strcmp(p->bctx[i], seed))
            {
                strncpy(fw[fn], p->bpred[i], MAX_WORD_LEN - 1);
                fw[fn][MAX_WORD_LEN - 1] = 0;
                fc[fn] = p->bcnt[i];
                fn++;
            }
        for (int a = 0; a < fn - 1; a++)
            for (int b = 0; b < fn - a - 1; b++)
                if (fc[b] < fc[b + 1])
                {
                    int t2 = fc[b];
                    fc[b] = fc[b + 1];
                    fc[b + 1] = t2;
                    char tw[MAX_WORD_LEN];
                    memcpy(tw, fw[b], MAX_WORD_LEN);
                    memcpy(fw[b], fw[b + 1], MAX_WORD_LEN);
                    memcpy(fw[b + 1], tw, MAX_WORD_LEN);
                }
        for (int s = 0; s < STYLE_WORDS; s++)
        {
            const char *kw = (s < fn) ? fw[s] : seed;
            strncpy(p->keywords[p->topic_count][s], kw, MAX_WORD_LEN - 1);
            p->keywords[p->topic_count][s][MAX_WORD_LEN - 1] = 0;
        }
        p->topic_count++;
    }
}

void train_from_file(Predictor *p, const char *fn)
{
    FILE *f = fopen(fn, "r");
    if (!f)
    {
        veryx_error("Cannot open file");
        return;
    }
    veryx_banner_mini();
    printf("\n  %s📖 N-gram training on:%s %s%s%s\n", YL, RS, CY, fn, RS);
    veryx_thinking("Analyzing corpus — hash-table engine active");
    scan_animation("INGESTING FILE DATA");
    data_stream(3);
    scan_animation("BUILDING HASH TABLES");
    char line[MAX_LINE];
    int lc = 0, tw = 0;
    while (fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\n")] = 0;
        if (*line)
        {
            /* Count words */
            char tmp[MAX_LINE];
            strncpy(tmp, line, sizeof(tmp) - 1);
            tmp[sizeof(tmp) - 1] = 0;
            char *w = strtok(tmp, " .,!?;:\n\t");
            while (w)
            {
                tw++;
                w = strtok(NULL, " .,!?;:\n\t");
            }
            train_from_string(p, line);
            lc++;
        }
        if (lc % 200 == 0)
            beep_train();
    }
    fclose(f);
    veryx_processing("REBUILDING FLAT ARRAYS");
    rebuild_flat(p);
    veryx_processing("DETECTING TOPIC PROFILES");
    detect_topics(p);
    printf("\n");
    box_top(GR);
    box_rowc(GR, "  " YL "⚡ VERYX" RS "  N-GRAM TRAINING COMPLETE");
    box_mid(GR);
    box_rowf(GR, "  📄 Lines: " GR "%d" RS "   📝 Words: " GR "%d" RS "   🎭 Topics: " YL "%d" RS, lc, tw, p->topic_count);
    box_rowf(GR, "  📚 Vocab: " GR "%d" RS "   🔗 Bigrams: " CY "%d" RS "   Trigrams: " CY "%d" RS, p->vocab_size, p->bigram_count, p->trigram_count);
    box_bot(GR);
    beep_success();
    auto_save(p);
    veryx_quip_for(1);
}

/* ============================================================
 * GENERATION HELPERS
 * ============================================================ */

/* Returns 1 if a word has appeared recently in the history array.
 * Used to suppress immediate repetition.                          */
static int _in_recent(const char history[][MAX_WORD_LEN], int hlen, const char *w)
{
    for (int i = 0; i < hlen; i++)
        if (!strcmp(history[i], w))
            return 1;
    return 0;
}

/* Weighted bigram pick with anti-repetition:
 * Tries up to RETRY times to avoid recently-used words.
 * Falls back to plain weighted pick if nothing else found.        */
#define GEN_RETRY 4
#define GEN_HIST 6
static int wb_norep(Predictor *p, const char *cw,
                    const char history[][MAX_WORD_LEN], int hlen,
                    char out[MAX_WORD_LEN])
{
    unsigned int h = djb2(cw) & (BIGRAM_HT_SIZE - 1);
    int total = 0;
    for (NgramNode *n = p->bht[h]; n; n = n->next)
        if (!strcmp(n->ctx, cw))
            total += n->count;
    if (!total)
        return 0;
    /* Try a few times to land on a non-repeated word */
    for (int attempt = 0; attempt < GEN_RETRY; attempt++)
    {
        int r = (rand() % total) + 1, a = 0;
        for (NgramNode *n = p->bht[h]; n; n = n->next)
            if (!strcmp(n->ctx, cw))
            {
                a += n->count;
                if (a >= r)
                {
                    /* On last attempt accept anything */
                    if (attempt == GEN_RETRY - 1 || !_in_recent(history, hlen, n->pred))
                    {
                        strncpy(out, n->pred, MAX_WORD_LEN - 1);
                        out[MAX_WORD_LEN - 1] = 0;
                        return 1;
                    }
                    break; /* retry */
                }
            }
    }
    return 0;
}

/* Weighted trigram pick (no repetition check — trigrams are specific enough) */
static int wt_pick(Predictor *p, const char *ctx, char out[MAX_WORD_LEN])
{
    unsigned int h = djb2(ctx) & (TRIGRAM_HT_SIZE - 1);
    int total = 0;
    for (NgramNode *n = p->tht[h]; n; n = n->next)
        if (!strcmp(n->ctx, ctx))
            total += n->count;
    if (!total)
        return 0;
    int r = (rand() % total) + 1, a = 0;
    for (NgramNode *n = p->tht[h]; n; n = n->next)
        if (!strcmp(n->ctx, ctx))
        {
            a += n->count;
            if (a >= r)
            {
                strncpy(out, n->pred, MAX_WORD_LEN - 1);
                out[MAX_WORD_LEN - 1] = 0;
                return 1;
            }
        }
    return 0;
}

/* Pick a random word from vocab weighted by frequency.
 * Used as a smart fallback instead of pure rand() % vocab_size.  */
static int wb_vocab_fallback(Predictor *p, char out[MAX_WORD_LEN])
{
    if (p->vocab_size <= 0)
        return 0;
    /* Build total weighted by count */
    long total = 0;
    for (int i = 0; i < p->vocab_size; i++)
        total += p->vcnt[i];
    if (!total)
        return 0;
    long r = (long)(rand() % total) + 1, a = 0;
    for (int i = 0; i < p->vocab_size; i++)
    {
        a += p->vcnt[i];
        if (a >= r)
        {
            strncpy(out, p->vocab[i], MAX_WORD_LEN - 1);
            out[MAX_WORD_LEN - 1] = 0;
            return 1;
        }
    }
    /* Fallback to last entry */
    strncpy(out, p->vocab[p->vocab_size - 1], MAX_WORD_LEN - 1);
    out[MAX_WORD_LEN - 1] = 0;
    return 1;
}

/* Natural sentence end detection:
 * Returns 1 if we should stop generating (natural boundary reached). */
static int _natural_end(const char *w, int wc, int min_words)
{
    if (wc < min_words)
        return 0;
    /* Hard stops */
    if (!strcmp(w, ".") || !strcmp(w, "!") || !strcmp(w, "?"))
        return 1;
    /* Soft stop: sentence-ending words after a reasonable length */
    if (wc >= 12)
    {
        static const char *soft[] = {
            "the", "a", "an", "and", "but", "or", "so",
            "because", "when", "while", "after", "before",
            "if", "although", "however", "therefore"};
        int ns = (int)(sizeof(soft) / sizeof(soft[0]));
        for (int i = 0; i < ns; i++)
            if (!strcmp(w, soft[i]))
                /* 40% chance to stop at a conjunction/article after word 12 */
                if (rand() % 5 < 2)
                    return 1;
    }
    return 0;
}

/* ============================================================
 * AUTO-GENERATE SENTENCE FROM A SEED WORD
 * Improvements over original:
 *  1. Anti-repetition history (GEN_HIST words)
 *  2. Temperature: high-count words boosted early, varied later
 *  3. Natural sentence endings via _natural_end()
 *  4. Smart fallback: if bigram dead-end, try a topic keyword
 *     then frequency-weighted vocab — not pure random
 *  5. Loop detection: if stuck in same word twice, force escape
 * ============================================================ */
char *generate_ngram_sentence(Predictor *p, const char *sw, int mw)
{
    static char S[MAX_LINE * 2];
    S[0] = 0;
    int max_words = (mw > 0) ? mw : MAX_NATURAL_WORDS;
    int min_words = (mw > 0) ? mw : 6; /* don't end too early in natural mode */

    char cw[MAX_WORD_LEN], pw[MAX_WORD_LEN] = "";
    char ctx[MAX_WORD_LEN * 2], nw[MAX_WORD_LEN];
    char history[GEN_HIST][MAX_WORD_LEN];
    int hlen = 0;

    strncpy(cw, sw, MAX_WORD_LEN - 1);
    cw[MAX_WORD_LEN - 1] = 0;
    to_lowercase(cw);

    /* Seed word into history and output */
    strncpy(S, cw, sizeof(S) - 1);
    S[sizeof(S) - 1] = 0;
    strncpy(history[hlen % GEN_HIST], cw, MAX_WORD_LEN - 1);
    history[hlen % GEN_HIST][MAX_WORD_LEN - 1] = 0;
    hlen++;

    printf("\n%s⚡ VERYX%s  %s▶%s %s%s%s", YL, RS, PU, RS, GR, S, RS);
    fflush(stdout);

    int stall = 0; /* counts consecutive times we got the same word */
    char last_added[MAX_WORD_LEN] = "";

    for (int wc = 1; wc < max_words; wc++)
    {
        int got = 0;

        /* 1. Try trigram first (most specific) */
        if (*pw)
        {
            snprintf(ctx, sizeof(ctx), "%s %s", pw, cw);
            got = wt_pick(p, ctx, nw);
        }

        /* 2. Try bigram with anti-repetition */
        if (!got)
            got = wb_norep(p, cw, (const char (*)[MAX_WORD_LEN])history, hlen < GEN_HIST ? hlen : GEN_HIST, nw);

        /* 3. Dead-end escape: backtrack to prev word's bigram */
        if (!got && *pw)
            got = wb_norep(p, pw, (const char (*)[MAX_WORD_LEN])history, hlen < GEN_HIST ? hlen : GEN_HIST, nw);

        /* 4. Frequency-weighted vocab fallback (not pure random) */
        if (!got)
            got = wb_vocab_fallback(p, nw);

        if (!got)
            break;

        /* Loop detection: same word picked 2x in a row → force vocab fallback */
        if (!strcmp(nw, last_added))
        {
            stall++;
            if (stall >= 2)
            {
                if (!wb_vocab_fallback(p, nw))
                    break;
                stall = 0;
            }
        }
        else
            stall = 0;
        strncpy(last_added, nw, MAX_WORD_LEN - 1);
        last_added[MAX_WORD_LEN - 1] = 0;

        /* Append to output */
        strncat(S, " ", sizeof(S) - strlen(S) - 1);
        strncat(S, nw, sizeof(S) - strlen(S) - 1);
        printf(" ");
        type_colored(nw, GR, 16);
        Beep(700 + (wc % 5) * 60, 12);

        /* Advance state */
        strncpy(pw, cw, MAX_WORD_LEN - 1);
        pw[MAX_WORD_LEN - 1] = 0;
        strncpy(cw, nw, MAX_WORD_LEN - 1);
        cw[MAX_WORD_LEN - 1] = 0;
        strncpy(history[hlen % GEN_HIST], nw, MAX_WORD_LEN - 1);
        history[hlen % GEN_HIST][MAX_WORD_LEN - 1] = 0;
        hlen++;

        if (_natural_end(nw, wc, min_words))
            break;
    }
    printf("\n");
    return S;
}

/* ============================================================
 * STYLE MIMIC SENTENCE GENERATION
 * Improvements over original:
 *  1. Seed from topic keywords in rotation (not always topic word)
 *     → sentences don't all start the same way
 *  2. Topic re-anchoring: every ~5 words nudge back toward topic
 *     keywords to keep sentences on-topic
 *  3. Anti-repetition history same as ngram version
 *  4. Frequency-weighted fallback instead of pure rand()
 *  5. Natural sentence endings
 *  6. Loop detection
 * ============================================================ */
char *generate_style_sentence(Predictor *p, const char *topic, int mw)
{
    static char S[MAX_LINE * 2];
    S[0] = 0;

    /* Find topic index */
    int ti = -1;
    for (int i = 0; i < p->topic_count; i++)
        if (!strcmp(p->topics[i], topic))
        {
            ti = i;
            break;
        }
    if (ti < 0)
    {
        snprintf(S, sizeof(S), "Topic '%s' not found.", topic);
        return S;
    }

    int max_words = (mw > 0) ? mw : MAX_NATURAL_WORDS;
    int min_words = (mw > 0) ? mw : 6;

    /* Collect non-empty keywords for this topic */
    char kws[STYLE_WORDS][MAX_WORD_LEN];
    int nkw = 0;
    for (int k = 0; k < STYLE_WORDS; k++)
        if (p->keywords[ti][k][0] != '\0')
        {
            strncpy(kws[nkw], p->keywords[ti][k], MAX_WORD_LEN - 1);
            kws[nkw][MAX_WORD_LEN - 1] = 0;
            nkw++;
        }
    /* Always include the topic word itself */
    if (nkw == 0)
    {
        strncpy(kws[0], p->topics[ti], MAX_WORD_LEN - 1);
        kws[0][MAX_WORD_LEN - 1] = 0;
        nkw = 1;
    }

    /* Choose starting word: rotate through keywords so sentences vary */
    static int _kw_rotate = 0;
    char cw[MAX_WORD_LEN];
    strncpy(cw, kws[_kw_rotate % nkw], MAX_WORD_LEN - 1);
    cw[MAX_WORD_LEN - 1] = 0;
    _kw_rotate++;

    char pw[MAX_WORD_LEN] = "";
    char ctx[MAX_WORD_LEN * 2], nw[MAX_WORD_LEN];
    char history[GEN_HIST][MAX_WORD_LEN];
    int hlen = 0;

    strncpy(S, cw, sizeof(S) - 1);
    S[sizeof(S) - 1] = 0;
    strncpy(history[hlen % GEN_HIST], cw, MAX_WORD_LEN - 1);
    history[hlen % GEN_HIST][MAX_WORD_LEN - 1] = 0;
    hlen++;

    printf("\n%s⚡ VERYX%s  %s🎭%s %s%s%s  %s▶%s %s%s%s",
           YL, RS, PU, RS, CY, topic, RS, PU, RS, GR, S, RS);
    fflush(stdout);

    int stall = 0;
    char last_added[MAX_WORD_LEN] = "";
    /* next word index after which we attempt a topic re-anchor */
    int next_anchor = 4 + rand() % 3; /* anchor around word 4-6 */

    for (int wc = 1; wc < max_words; wc++)
    {
        int got = 0;

        /* Topic re-anchoring: occasionally inject a topic keyword as the
         * next context so the sentence stays on theme                    */
        if (wc == next_anchor && nkw > 0)
        {
            /* Try to find a bigram continuation from a keyword */
            for (int k = 0; k < nkw && !got; k++)
            {
                int ki = (k + _kw_rotate) % nkw;
                if (!_in_recent(history, hlen < GEN_HIST ? hlen : GEN_HIST, kws[ki]))
                    got = wb_norep(p, kws[ki],
                                   (const char (*)[MAX_WORD_LEN])history,
                                   hlen < GEN_HIST ? hlen : GEN_HIST, nw);
                if (got)
                {
                    /* Insert the keyword itself before its continuation */
                    strncat(S, " ", sizeof(S) - strlen(S) - 1);
                    strncat(S, kws[ki], sizeof(S) - strlen(S) - 1);
                    printf(" ");
                    type_colored(kws[ki], CY, 16);
                    Beep(800, 12);
                    strncpy(pw, cw, MAX_WORD_LEN - 1);
                    pw[MAX_WORD_LEN - 1] = 0;
                    strncpy(cw, kws[ki], MAX_WORD_LEN - 1);
                    cw[MAX_WORD_LEN - 1] = 0;
                    strncpy(history[hlen % GEN_HIST], kws[ki], MAX_WORD_LEN - 1);
                    history[hlen % GEN_HIST][MAX_WORD_LEN - 1] = 0;
                    hlen++;
                    wc++;
                    if (wc >= max_words)
                        goto done;
                    break;
                }
            }
            next_anchor = wc + 4 + rand() % 4; /* schedule next anchor */
        }

        /* 1. Trigram */
        if (*pw)
        {
            snprintf(ctx, sizeof(ctx), "%s %s", pw, cw);
            got = wt_pick(p, ctx, nw);
        }

        /* 2. Bigram with anti-repetition */
        if (!got)
            got = wb_norep(p, cw,
                           (const char (*)[MAX_WORD_LEN])history,
                           hlen < GEN_HIST ? hlen : GEN_HIST, nw);

        /* 3. Backtrack to prev word */
        if (!got && *pw)
            got = wb_norep(p, pw,
                           (const char (*)[MAX_WORD_LEN])history,
                           hlen < GEN_HIST ? hlen : GEN_HIST, nw);

        /* 4. Frequency-weighted vocab fallback */
        if (!got)
            got = wb_vocab_fallback(p, nw);

        if (!got)
            break;

        /* Loop detection */
        if (!strcmp(nw, last_added))
        {
            stall++;
            if (stall >= 2)
            {
                if (!wb_vocab_fallback(p, nw))
                    break;
                stall = 0;
            }
        }
        else
            stall = 0;
        strncpy(last_added, nw, MAX_WORD_LEN - 1);
        last_added[MAX_WORD_LEN - 1] = 0;

        strncat(S, " ", sizeof(S) - strlen(S) - 1);
        strncat(S, nw, sizeof(S) - strlen(S) - 1);
        printf(" ");
        type_colored(nw, GR, 16);
        Beep(800 + (wc % 4) * 70, 12);

        strncpy(pw, cw, MAX_WORD_LEN - 1);
        pw[MAX_WORD_LEN - 1] = 0;
        strncpy(cw, nw, MAX_WORD_LEN - 1);
        cw[MAX_WORD_LEN - 1] = 0;
        strncpy(history[hlen % GEN_HIST], nw, MAX_WORD_LEN - 1);
        history[hlen % GEN_HIST][MAX_WORD_LEN - 1] = 0;
        hlen++;

        if (_natural_end(nw, wc, min_words))
            break;
    }
done:
    printf("\n");
    return S;
}

/* ============================================================
 * FAST PREDICT_NEXT_WORDS — uses hash tables directly
 * Trigram and bigram lookups are now O(chain) not O(all_ngrams)
 * ============================================================ */
int predict_next_words(Predictor *p, const char *w1, const char *w2, Prediction res[])
{
    char ctx[MAX_WORD_LEN * 3] = "", w1l[MAX_WORD_LEN] = "", w2l[MAX_WORD_LEN] = "";
    if (*w1)
    {
        strncpy(w1l, w1, MAX_WORD_LEN - 1);
        w1l[MAX_WORD_LEN - 1] = 0;
        to_lowercase(w1l);
    }
    if (*w2)
    {
        strncpy(w2l, w2, MAX_WORD_LEN - 1);
        w2l[MAX_WORD_LEN - 1] = 0;
        to_lowercase(w2l);
    }
    int mt = p->trigram_count + p->bigram_count + 1;
    Prediction *tmp = (Prediction *)malloc((size_t)mt * sizeof(Prediction));
    if (!tmp)
        return 0;
    int tc = 0;

    /* ── Trigram lookup via hash table ── */
    if (*w1l && *w2l)
    {
        snprintf(ctx, sizeof(ctx), "%s %s", w1l, w2l);
        unsigned int th = djb2(ctx) & (TRIGRAM_HT_SIZE - 1);
        /* first pass: sum totals for confidence calculation */
        float ttotal = 0;
        for (NgramNode *n = p->tht[th]; n; n = n->next)
            if (!strcmp(n->ctx, ctx))
                ttotal += (float)n->count;
        /* second pass: collect predictions */
        for (NgramNode *n = p->tht[th]; n && tc < mt; n = n->next)
            if (!strcmp(n->ctx, ctx))
            {
                strncpy(tmp[tc].word, n->pred, MAX_WORD_LEN - 1);
                tmp[tc].word[MAX_WORD_LEN - 1] = 0;
                tmp[tc].count = n->count;
                tmp[tc].confidence = (ttotal > 0) ? ((float)n->count / ttotal) * 100.0f : 0.0f;
                tc++;
            }
    }

    /* ── Bigram lookup via hash table ── */
    if (*w2l)
    {
        unsigned int bh = djb2(w2l) & (BIGRAM_HT_SIZE - 1);
        /* first pass: sum totals for confidence calculation */
        float btotal = 0;
        for (NgramNode *n = p->bht[bh]; n; n = n->next)
            if (!strcmp(n->ctx, w2l))
                btotal += (float)n->count;
        /* second pass: collect predictions (skip duplicates from trigram) */
        for (NgramNode *n = p->bht[bh]; n && tc < mt; n = n->next)
            if (!strcmp(n->ctx, w2l))
            {
                int already = 0;
                for (int j = 0; j < tc; j++)
                    if (!strcmp(tmp[j].word, n->pred))
                    {
                        already = 1;
                        break;
                    }
                if (!already)
                {
                    strncpy(tmp[tc].word, n->pred, MAX_WORD_LEN - 1);
                    tmp[tc].word[MAX_WORD_LEN - 1] = 0;
                    tmp[tc].count = n->count;
                    tmp[tc].confidence = (btotal > 0) ? ((float)n->count / btotal) * 100.0f : 0.0f;
                    tc++;
                }
            }
    }

    if (tc > 0)
    {
        for (int i = 0; i < tc - 1; i++)
            for (int j = 0; j < tc - i - 1; j++)
                if (tmp[j].count < tmp[j + 1].count)
                {
                    Prediction t2 = tmp[j];
                    tmp[j] = tmp[j + 1];
                    tmp[j + 1] = t2;
                }
        int nr = (tc < TOP_K) ? tc : TOP_K;
        for (int i = 0; i < nr; i++)
            res[i] = tmp[i];
        free(tmp);
        return nr;
    }
    free(tmp);
    return 0;
}

void print_predictions(Prediction pred[], int count)
{
    if (!count)
    {
        printf("\n");
        box_top(PU);
        box_rowc(PU, "  " YL "⚡ VERYX" RS "  " PU "🔮 NEXT WORD PREDICTIONS" RS);
        box_mid(PU);
        box_rowc(PU, "  " DM "No predictions available for this context." RS);
        box_bot(PU);
        return;
    }

    printf("\n");
    box_top(PU);
    /* ── Header ─────────────────────────────────────────── */
    box_centerc(PU, YL "⚡ VERYX" RS "  " PU "🔮  NEXT WORD  PREDICTIONS" RS);
    box_mid(PU);

    /* "X suggestions found" row */
    {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "  🔍  Showing top " GR "%d" RS " suggestion%s  " DM "— ranked by frequency" RS,
                 count, count == 1 ? "" : "s");
        box_rowc(PU, buf);
    }

    box_sep(PU);

    /* Column header row */
    box_rowc(PU,
             "  " DM
             "Rank        Word              Accuracy Bar"
             "              Conf    Freq" RS);

    box_sep(PU);

    /* ── Result rows ──────────────────────────────────────── */
    for (int i = 0; i < count; i++)
    {

        /* Badge: ★ BEST / 2nd / 3rd */
        const char *badge_col = (i == 0) ? YL : (i == 1) ? CY
                                                         : WH;
        const char *badge = (i == 0) ? " " YL "★ BEST" RS " " : (i == 1) ? " " CY "  2nd " RS " "
                                                                         : " " WH "  3rd " RS " ";

        /* Word colour by rank */
        const char *wc = (i == 0) ? GR : (i == 1) ? YL
                                     : (i == 2)   ? CY
                                                  : WH;

        /* Accuracy bar — 20 chars wide, based on confidence 0-100 */
        int filled = (int)(pred[i].confidence / 5.0f); /* 100% → 20 */
        if (filled > 20)
            filled = 20;
        if (filled < 0)
            filled = 0;

        /* Bar colour */
        const char *bar_col = (i == 0) ? GR : (i == 1) ? YL
                                                       : DM;

        /* Build full coloured row */
        char row[512];
        snprintf(row, sizeof(row),
                 " %s  %s%d.%s  %s%-16.16s%s  %s[",
                 badge,
                 badge_col, (i + 1), RS,
                 wc, pred[i].word, RS,
                 bar_col);

        /* Append bar characters */
        char bar_chars[256] = {0};
        for (int b = 0; b < 20; b++)
        {
            const char *blk = (b < filled) ? "\xe2\x96\x88" : "\xe2\x96\x91";
            strncat(bar_chars, blk, sizeof(bar_chars) - strlen(bar_chars) - 1);
        }

        /* Append rest of row */
        char row2[256];
        snprintf(row2, sizeof(row2),
                 "%s%s]%s  " CY "%5.1f%%" RS "  " PU "%d" RS,
                 bar_chars, bar_col, RS,
                 pred[i].confidence,
                 pred[i].count);

        /* Merge and print */
        char full_row[768];
        snprintf(full_row, sizeof(full_row), "%s%s", row, row2);
        box_rowc(PU, full_row);

        if (i < count - 1)
            box_sep(PU);
        Beep(900 - i * 80, 25);
        Sleep(20);
    }

    /* ── Best match footer ────────────────────────────────── */
    box_mid(PU);
    {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "  ⭐  Best suggestion:  " GR "%-16s" RS
                 "  (conf: " CY "%.1f%%" RS "  freq: " PU "%d" RS ")",
                 pred[0].word, pred[0].confidence, pred[0].count);
        box_row(PU, buf);
    }
    box_bot(PU);
}

void print_stats(Predictor *p)
{
    veryx_banner_mini();
    veryx_processing("COMPILING N-GRAM MODEL STATISTICS");
    printf("\n");
    box_top(YL);
    box_rowc(YL, "  " YL "📊 N-GRAM MODEL STATISTICS" RS);
    box_mid(YL);
    box_rowf(YL, "  📚 Vocabulary  : " GR "%d" RS " unique words", p->vocab_size);
    box_rowf(YL, "  🔗 Bigrams     : " CY "%d" RS " sequences", p->bigram_count);
    box_rowf(YL, "  🔗 Trigrams    : " CY "%d" RS " sequences", p->trigram_count);
    box_rowf(YL, "  📈 Bigram occ  : " PU "%d" RS, p->total_bigrams);
    box_rowf(YL, "  📈 Trigram occ : " PU "%d" RS, p->total_trigrams);
    if (p->topic_count > 0)
    {
        box_mid(YL);
        box_rowf(YL, "  Detected Topics (%d):", p->topic_count);
        box_sep(YL);
        for (int i = 0; i < p->topic_count; i++)
        {
            char line[TERM_W + 1];
            char kws[40] = "";
            int kc = 0;
            for (int k = 0; k < STYLE_WORDS && kc < 3; k++)
                if (*p->keywords[i][k] && strcmp(p->keywords[i][k], p->topics[i]))
                {
                    if (kc)
                        strcat(kws, ", ");
                    strncat(kws, p->keywords[i][k], 39 - (int)strlen(kws));
                    kc++;
                }
            snprintf(line, sizeof(line), "  %2d. %-12s  -> %s", i + 1, p->topics[i], kws);
            box_row(YL, line);
        }
    }
    box_bot(YL);
}

void save_model(Predictor *p, const char *fn)
{
    FILE *f = fopen(fn, "w");
    if (!f)
    {
        veryx_error("Cannot save model");
        return;
    }
    fprintf(f, "VOCAB:%d\n", p->vocab_size);
    for (int i = 0; i < p->vocab_size; i++)
        fprintf(f, "%s:%d\n", p->vocab[i], p->vcnt[i]);
    fprintf(f, "BIGRAMS:%d\n", p->bigram_count);
    for (int i = 0; i < p->bigram_count; i++)
        fprintf(f, "%s|%s:%d\n", p->bctx[i], p->bpred[i], p->bcnt[i]);
    fprintf(f, "TRIGRAMS:%d\n", p->trigram_count);
    for (int i = 0; i < p->trigram_count; i++)
        fprintf(f, "%s|%s:%d\n", p->tctx[i], p->tpred[i], p->tcnt[i]);
    fprintf(f, "TOTALS:%d:%d\n", p->total_bigrams, p->total_trigrams);
    fclose(f);
}

void load_model(Predictor *p, const char *fn)
{
    FILE *f = fopen(fn, "r");
    if (!f)
    {
        printf("\n  %s⚡ VERYX%s  No saved N-gram model.\n", YL, RS);
        return;
    }
    /* Large read buffer — reduces fread syscalls dramatically */
    setvbuf(f, NULL, _IOFBF, 1 << 20); /* 1 MB read buffer */
    veryx_processing("LOADING N-GRAM MODEL FROM DISK");
    char line[MAX_LINE];
    free_predictor(p);
    init_predictor(p);

    /* ── VOCAB ── */
    int vs = 0;
    if (!fgets(line, sizeof(line), f))
    {
        fclose(f);
        return;
    }
    if (sscanf(line, "VOCAB:%d", &vs) != 1)
    {
        fclose(f);
        return;
    }
    for (int i = 0; i < vs; i++)
    {
        if (!fgets(line, sizeof(line), f))
            break;
        line[strcspn(line, "\n")] = 0;
        char *c = strrchr(line, ':');
        if (!c)
            continue;
        *c = 0;
        /* Direct insert — no duplicate check needed when loading saved file */
        unsigned int h = djb2(line) & (VOCAB_HT_SIZE - 1);
        VocabNode *nd = (VocabNode *)malloc(sizeof(VocabNode));
        if (!nd)
            continue;
        strncpy(nd->key, line, MAX_WORD_LEN - 1);
        nd->key[MAX_WORD_LEN - 1] = 0;
        nd->count = atoi(c + 1);
        nd->next = p->vht[h];
        p->vht[h] = nd;
    }

    /* ── BIGRAMS ── */
    p->total_bigrams = 0;
    int bc = 0;
    if (!fgets(line, sizeof(line), f))
    {
        fclose(f);
        goto ld;
    }
    if (sscanf(line, "BIGRAMS:%d", &bc) != 1)
    {
        fclose(f);
        goto ld;
    }
    for (int i = 0; i < bc; i++)
    {
        if (!fgets(line, sizeof(line), f))
            break;
        line[strcspn(line, "\n")] = 0;
        char *pipe = strchr(line, '|');
        if (!pipe)
            continue;
        char *col = strrchr(pipe, ':');
        if (!col)
            continue;
        *pipe = 0;
        *col = 0;
        int cnt = atoi(col + 1);
        /* Direct insert — hash by ctx only, set count directly, no chain walk */
        unsigned int h = djb2(line) & (BIGRAM_HT_SIZE - 1);
        NgramNode *nd = (NgramNode *)malloc(sizeof(NgramNode));
        if (!nd)
            continue;
        strncpy(nd->ctx, line, MAX_WORD_LEN * 3 - 1);
        nd->ctx[MAX_WORD_LEN * 3 - 1] = 0;
        strncpy(nd->pred, pipe + 1, MAX_WORD_LEN - 1);
        nd->pred[MAX_WORD_LEN - 1] = 0;
        nd->count = cnt;
        nd->next = p->bht[h];
        p->bht[h] = nd;
        p->total_bigrams += cnt;
    }

    /* ── TRIGRAMS ── */
    p->total_trigrams = 0;
    int tc = 0;
    if (!fgets(line, sizeof(line), f))
    {
        fclose(f);
        goto ld;
    }
    if (sscanf(line, "TRIGRAMS:%d", &tc) != 1)
    {
        fclose(f);
        goto ld;
    }
    for (int i = 0; i < tc; i++)
    {
        if (!fgets(line, sizeof(line), f))
            break;
        line[strcspn(line, "\n")] = 0;
        char *pipe = strchr(line, '|');
        if (!pipe)
            continue;
        char *col = strrchr(pipe, ':');
        if (!col)
            continue;
        *pipe = 0;
        *col = 0;
        int cnt = atoi(col + 1);
        /* Direct insert — hash by ctx only, set count directly, no chain walk */
        unsigned int h = djb2(line) & (TRIGRAM_HT_SIZE - 1);
        NgramNode *nd = (NgramNode *)malloc(sizeof(NgramNode));
        if (!nd)
            continue;
        strncpy(nd->ctx, line, MAX_WORD_LEN * 3 - 1);
        nd->ctx[MAX_WORD_LEN * 3 - 1] = 0;
        strncpy(nd->pred, pipe + 1, MAX_WORD_LEN - 1);
        nd->pred[MAX_WORD_LEN - 1] = 0;
        nd->count = cnt;
        nd->next = p->tht[h];
        p->tht[h] = nd;
        p->total_trigrams += cnt;
    }

    if (fgets(line, sizeof(line), f))
        sscanf(line, "TOTALS:%d:%d", &p->total_bigrams, &p->total_trigrams);
ld:
    fclose(f);
    rebuild_flat(p);
    detect_topics(p);
    printf("\n  %s⚡ VERYX%s  %s✅ N-gram loaded%s — vocab:%s%d%s  bigrams:%s%d%s  topics:%s%d%s\n",
           GR, RS, GR, RS, CY, p->vocab_size, RS, CY, p->bigram_count, RS, YL, p->topic_count, RS);
    beep_success();
}
void auto_save(Predictor *p)
{
    save_model(p, MODEL_FILE);
    veryx_success("N-gram model auto-saved");
}
void auto_load(Predictor *p) { load_model(p, MODEL_FILE); }
void reset_model(Predictor *p)
{
    veryx_processing("WIPING N-GRAM DATA");
    free_predictor(p);
    init_predictor(p);
    remove(MODEL_FILE);
    veryx_success("N-gram model reset");
}

// ============================================================
// SPELL CORRECTOR — hash-table backed
// ============================================================

void init_spell(SpellCorrector *sc)
{
    sc->sht = (SpellNode **)calloc(SPELL_HT_SIZE, sizeof(SpellNode *));
    if (!sc->sht)
    {
        veryx_error("spell alloc");
        exit(1);
    }
    sc->dict_size = sc->total_words = 0;
}
void free_spell(SpellCorrector *sc)
{
    if (sc->sht)
    {
        for (int i = 0; i < SPELL_HT_SIZE; i++)
        {
            SpellNode *n = sc->sht[i];
            while (n)
            {
                SpellNode *nx = n->next;
                free(n);
                n = nx;
            }
        }
        free(sc->sht);
        sc->sht = NULL;
    }
    sc->dict_size = sc->total_words = 0;
}

static void spell_add(SpellCorrector *sc, const char *w)
{
    unsigned int h = djb2(w) & (SPELL_HT_SIZE - 1);
    sc->total_words++;
    for (SpellNode *n = sc->sht[h]; n; n = n->next)
        if (!strcmp(n->word, w))
        {
            n->freq++;
            return;
        }
    SpellNode *nd = (SpellNode *)malloc(sizeof(SpellNode));
    if (!nd)
        return;
    strncpy(nd->word, w, MAX_WORD_LEN - 1);
    nd->word[MAX_WORD_LEN - 1] = 0;
    nd->freq = 1;
    nd->next = sc->sht[h];
    sc->sht[h] = nd;
    sc->dict_size++;
}

void spell_train_from_file(SpellCorrector *sc, const char *fn)
{
    FILE *f = fopen(fn, "r");
    if (!f)
    {
        veryx_error("Cannot open spell file");
        return;
    }
    veryx_banner_mini();
    printf("\n  %s📖 Spell training on:%s %s%s%s\n", YL, RS, CY, fn, RS);
    veryx_thinking("Indexing word corpus — hash-table engine active");
    scan_animation("READING CORPUS");
    data_stream(3);
    scan_animation("INDEXING DICTIONARY");
    char line[MAX_LINE];
    int lc = 0;
    while (fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\n")] = 0;
        if (*line)
        {
            size_t tl = strlen(line);
            char *tc = (char *)malloc(tl + 1);
            if (!tc)
                continue;
            memcpy(tc, line, tl + 1);
            char *tok = strtok(tc, " .,!?;:\"'()\n\t-_0123456789");
            while (tok)
            {
                int al = 1;
                for (int i = 0; tok[i]; i++)
                    if (!isalpha((unsigned char)tok[i]))
                    {
                        al = 0;
                        break;
                    }
                if (al && strlen(tok) >= 2 && strlen(tok) < MAX_WORD_LEN)
                {
                    to_lowercase(tok);
                    spell_add(sc, tok);
                }
                tok = strtok(NULL, " .,!?;:\"'()\n\t-_0123456789");
            }
            free(tc);
            lc++;
        }
        if (lc % 200 == 0)
            beep_train();
    }
    fclose(f);
    printf("\n");
    box_top(CY);
    box_rowc(CY, "  " YL "⚡ VERYX" RS "  SPELL TRAINING COMPLETE");
    box_mid(CY);
    box_rowf(CY, "  📚 Unique: " GR "%d" RS "   Total: " CY "%d" RS, sc->dict_size, sc->total_words);
    box_bot(CY);
    beep_success();
    spell_auto_save(sc);
    veryx_quip_for(1);
}

void spell_save(SpellCorrector *sc, const char *fn)
{
    FILE *f = fopen(fn, "w");
    if (!f)
    {
        veryx_error("Cannot save spell model");
        return;
    }
    int total = 0;
    for (int i = 0; i < SPELL_HT_SIZE; i++)
        for (SpellNode *n = sc->sht[i]; n; n = n->next)
            total++;
    fprintf(f, "SPELL_DICT:%d\n", total);
    for (int i = 0; i < SPELL_HT_SIZE; i++)
        for (SpellNode *n = sc->sht[i]; n; n = n->next)
            fprintf(f, "%s:%d\n", n->word, n->freq);
    fprintf(f, "TOTAL:%d\n", sc->total_words);
    fclose(f);
}
void spell_load(SpellCorrector *sc, const char *fn)
{
    FILE *f = fopen(fn, "r");
    if (!f)
    {
        printf("\n  %s⚡ VERYX%s  No saved spell model.\n", YL, RS);
        return;
    }
    veryx_processing("LOADING SPELL DICTIONARY");
    char line[MAX_LINE];
    free_spell(sc);
    init_spell(sc);
    if (!fgets(line, sizeof(line), f))
    {
        fclose(f);
        return;
    }
    int ds = 0;
    if (sscanf(line, "SPELL_DICT:%d", &ds) != 1)
    {
        fclose(f);
        return;
    }
    for (int i = 0; i < ds; i++)
    {
        if (!fgets(line, sizeof(line), f))
            break;
        line[strcspn(line, "\n")] = 0;
        char *c = strrchr(line, ':');
        if (!c)
            continue;
        *c = 0;
        int fq = atoi(c + 1);
        spell_add(sc, line);
        unsigned int h = djb2(line) & (SPELL_HT_SIZE - 1);
        for (SpellNode *n = sc->sht[h]; n; n = n->next)
            if (!strcmp(n->word, line))
            {
                n->freq = fq;
                break;
            }
    }
    if (fgets(line, sizeof(line), f))
        sscanf(line, "TOTAL:%d", &sc->total_words);
    fclose(f);
    printf("\n  %s⚡ VERYX%s  Spell loaded — %s%d%s words\n", CY, RS, GR, sc->dict_size, RS);
    beep_success();
}
void spell_auto_save(SpellCorrector *sc)
{
    spell_save(sc, SPELL_MODEL_FILE);
    veryx_success("Spell model auto-saved");
}
void spell_auto_load(SpellCorrector *sc) { spell_load(sc, SPELL_MODEL_FILE); }
void spell_reset(SpellCorrector *sc)
{
    veryx_processing("CLEARING SPELL DICTIONARY");
    free_spell(sc);
    init_spell(sc);
    remove(SPELL_MODEL_FILE);
    veryx_success("Spell model reset");
}

void spell_print_stats(SpellCorrector *sc)
{
    printf("\n");
    box_top(CY);
    box_rowc(CY, "  " YL "📊 SPELL CORRECTOR STATISTICS" RS);
    box_mid(CY);
    box_rowf(CY, "  Unique words : %d", sc->dict_size);
    box_rowf(CY, "  Total tokens : %d", sc->total_words);
    box_rowf(CY, "  Max edit dist: %d", SPELL_MAX_DIST);
    box_rowf(CY, "  Top results  : %d", SPELL_TOP_K);
    box_bot(CY);
}

int levenshtein(const char *a, const char *b)
{
    int la = (int)strlen(a), lb = (int)strlen(b);
    if (!la)
        return lb;
    if (!lb)
        return la;
    if (la > lb)
    {
        const char *t = a;
        a = b;
        b = t;
        int tmp = la;
        la = lb;
        lb = tmp;
    }
    int *prev = (int *)malloc((la + 1) * sizeof(int)), *curr = (int *)malloc((la + 1) * sizeof(int));
    if (!prev || !curr)
    {
        free(prev);
        free(curr);
        return 9999;
    }
    for (int i = 0; i <= la; i++)
        prev[i] = i;
    for (int j = 1; j <= lb; j++)
    {
        curr[0] = j;
        for (int i = 1; i <= la; i++)
        {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int d = prev[i] + 1, in = curr[i - 1] + 1, s = prev[i - 1] + cost;
            curr[i] = d < in ? (d < s ? d : s) : (in < s ? in : s);
        }
        int *t = prev;
        prev = curr;
        curr = t;
    }
    int r = prev[la];
    free(prev);
    free(curr);
    return r;
}

int spell_correct(SpellCorrector *sc, const char *query, SpellCandidate res[])
{
    if (!sc->dict_size)
        return 0;
    char q[MAX_WORD_LEN];
    strncpy(q, query, MAX_WORD_LEN - 1);
    q[MAX_WORD_LEN - 1] = 0;
    to_lowercase(q);
    /* exact check via hash */
    unsigned int eh = djb2(q) & (SPELL_HT_SIZE - 1);
    for (SpellNode *n = sc->sht[eh]; n; n = n->next)
        if (!strcmp(n->word, q))
        {
            strncpy(res[0].word, n->word, MAX_WORD_LEN - 1);
            res[0].word[MAX_WORD_LEN - 1] = 0;
            res[0].dist = 0;
            res[0].freq = n->freq;
            res[0].score = 0;
            return 1;
        }
    int cap = 1024;
    SpellCandidate *cands = (SpellCandidate *)malloc(cap * sizeof(SpellCandidate));
    if (!cands)
        return 0;
    int nc = 0;
    for (int i = 0; i < SPELL_HT_SIZE; i++)
        for (SpellNode *n = sc->sht[i]; n; n = n->next)
        {
            int d = levenshtein(q, n->word);
            if (d > SPELL_MAX_DIST)
                continue;
            if (nc >= cap)
            {
                cap *= 2;
                SpellCandidate *t = (SpellCandidate *)realloc(cands, cap * sizeof(SpellCandidate));
                if (!t)
                {
                    free(cands);
                    return 0;
                }
                cands = t;
            }
            strncpy(cands[nc].word, n->word, MAX_WORD_LEN - 1);
            cands[nc].word[MAX_WORD_LEN - 1] = 0;
            cands[nc].dist = d;
            cands[nc].freq = n->freq;
            cands[nc].score = (float)d - (float)n->freq / (float)(sc->total_words + 1) * 0.5f;
            nc++;
        }
    if (!nc)
    {
        free(cands);
        return 0;
    }
    for (int i = 0; i < nc - 1; i++)
        for (int j = 0; j < nc - i - 1; j++)
            if (cands[j].score > cands[j + 1].score)
            {
                SpellCandidate t = cands[j];
                cands[j] = cands[j + 1];
                cands[j + 1] = t;
            }
    int ret = (nc < SPELL_TOP_K) ? nc : SPELL_TOP_K;
    for (int i = 0; i < ret; i++)
        res[i] = cands[i];
    free(cands);
    return ret;
}

void spell_correct_interactive(SpellCorrector *sc)
{
    print_banner();
    printf("\n");
    box_top(CY);
    box_rowc(CY, "  " YL "⚡ VERYX" RS "  " CY "🔤 SPELL CORRECTOR" RS);
    box_mid(CY);
    box_rowc(CY, "  " WH "Type a word to check." RS "  '" GR "quit" RS "' to return.");
    box_bot(CY);
    if (!sc->dict_size)
    {
        veryx_error("Dictionary empty! Admin must train it first.");
        wait_for_enter();
        return;
    }
    char input[MAX_WORD_LEN];
    while (1)
    {
        printf("\n  %s⚡ VERYX%s  %s🔤 Enter word:%s ", YL, RS, CY, RS);
        fflush(stdout);
        if (scanf("%49s", input) != 1)
        {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();
        if (!strcmp(input, "quit") || !strcmp(input, "exit"))
            break;
        char clean[MAX_WORD_LEN];
        int ci = 0;
        for (int i = 0; input[i] && ci < MAX_WORD_LEN - 1; i++)
            if (isalpha((unsigned char)input[i]))
                clean[ci++] = input[i];
        clean[ci] = 0;
        if (!ci)
        {
            veryx_error("Letters only please");
            continue;
        }
        veryx_maybe_remark(5); /* 20% chance before each spell check */
        printf("\n  %s⚡ VERYX%s  scanning dictionary...\n", YL, RS);
        neural_pulse(8);
        beep_spell();
        SpellCandidate res[SPELL_TOP_K];
        int n = spell_correct(sc, clean, res);
        if (!n)
        {
            veryx_error("No suggestions found");
            continue;
        }
        if (n == 1 && res[0].dist == 0)
        {
            printf("\n");
            box_top(GR);
            box_rowf(GR, "  CORRECT!  '%s' is spelled correctly  (freq: %d)", res[0].word, res[0].freq);
            box_bot(GR);
            beep_success();
            continue;
        }
        printf("\n");
        box_top(CY);
        box_centerc(CY, YL "⚡ VERYX" RS "  " CY "🔤  SPELL CHECKER  RESULTS" RS);
        box_mid(CY);
        /* typed word row */
        {
            char b[128];
            snprintf(b, sizeof(b), "  ❓  You typed:  " RD "%s" RS "    →  " GR "%d" RS " suggestion%s found",
                     clean, n, n == 1 ? "" : "s");
            box_rowc(CY, b);
        }
        box_sep(CY);
        /* column header */
        {
            char hdr[TERM_W + 1];
            snprintf(hdr, sizeof(hdr), "  %-6s  %-4s  %-18s  %-8s  %-6s  %s",
                     "Rank", "", "Word", "Accuracy", "Dist", "Freq");
            box_rowc(CY, DM "  Rank      Word               Accuracy Bar          Dist   Freq" RS);
        }
        box_sep(CY);
        /* result rows */
        for (int i = 0; i < n; i++)
        {
            /* colour the word by rank */
            const char *wc2 = (i == 0) ? GR : (i == 1) ? YL
                                          : (i == 2)   ? CY
                                                       : WH;
            const char *badge = (i == 0) ? " " YL "★ BEST" RS " " : (i == 1) ? " " CY "  2nd " RS " "
                                                                : (i == 2)   ? " " WH "  3rd " RS " "
                                                                             : "       ";
            /* accuracy 0-100 based on inverse edit distance */
            int acc_pct = (res[i].dist == 0) ? 100 : (res[i].dist == 1) ? 85
                                                 : (res[i].dist == 2)   ? 60
                                                                        : 35;
            /* bar: 20 chars wide */
            int filled = (acc_pct * 20) / 100;
            char bar[64] = "[";
            for (int b2 = 0; b2 < 20; b2++)
                strcat(bar, b2 < filled ? "█" : "░");
            strcat(bar, "]");
            /* compose coloured row */
            char row[512];
            snprintf(row, sizeof(row),
                     " %s  %s%d.%s  %s%-16.16s%s  %s%s%s  " DM "d=%d" RS "  " PU "%d" RS,
                     badge,
                     GR, (i + 1), RS,
                     wc2, res[i].word, RS,
                     (i == 0) ? GR : (i == 1) ? YL
                                              : DM,
                     bar, RS,
                     res[i].dist, res[i].freq);
            box_rowc(CY, row);
            if (i < n - 1)
                box_sep(CY);
            Beep(1000 - i * 60, 25);
            Sleep(20);
        }
        box_mid(CY);
        /* best suggestion highlight */
        {
            char b[256];
            snprintf(b, sizeof(b), "  ⭐  Best match:  " GR "%s" RS "   (edit-dist: " YL "%d" RS "  freq: " PU "%d" RS ")",
                     res[0].word, res[0].dist, res[0].freq);
            box_rowc(CY, b);
        }
        box_bot(CY);
        printf("\n  %s⚡ VERYX suggests:%s  ", YL, RS);
        type_colored(res[0].word, GR, 30);
        printf("\n");
        beep_spell();
        if (rand() % 3 == 0)
            veryx_quip_for(3); /* 1-in-3 chance */
    }
}

// ============================================================
// ============================================================
// VERYX GAME ARCADE  v2  —  Timer · Levels · Spell-Check · Hints
// ============================================================

/* ── Level config ─────────────────────────────────────────── */
typedef struct
{
    const char *name; /* "Easy" / "Medium" / "Hard" */
    const char *color;
    int rounds;       /* number of rounds            */
    int min_word_len; /* minimum word length to pick */
    int time_limit;   /* seconds per round (0 = none)*/
    int max_attempts; /* attempts per round (scramble)*/
    int pts_first;    /* pts for fastest/best answer  */
    int pts_second;
    int pts_third;
} Level;

static const Level LEVELS[3] = {
    /*  name    color  rounds  minlen  timer  maxatt  pts1 pts2 pts3 */
    {"Easy", GR, 4, 3, 0, 3, 3, 2, 1},    /* 3 attempts, no timer */
    {"Medium", YL, 6, 5, 20, 3, 3, 2, 0}, /* 3 attempts, 20s timer */
    {"Hard", RD, 8, 7, 12, 3, 3, 0, 0},   /* 3 attempts, 12s timer */
};

/* ── Timer helpers ────────────────────────────────────────── */

/* Returns seconds elapsed since start_t */
static double elapsed(time_t start_t)
{
    return difftime(time(NULL), start_t);
}

/* Draw a one-line live timer bar inside a box row.
   Called BEFORE reading input to show the limit.            */
static void show_timer_row(const char *bc, int limit_sec)
{
    if (limit_sec <= 0)
        return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "  ⏱️   Time limit: " YL "%d seconds" RS "  — Type fast!",
             limit_sec);
    box_rowc(bc, buf);
}

/* ── Shared helpers ───────────────────────────────────────── */

/* Fisher-Yates shuffle */
static void shuffle_str(char *s, int len)
{
    for (int i = len - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        char t = s[i];
        s[i] = s[j];
        s[j] = t;
    }
}

/* Pick vocab word of length >= min_len */
static int pick_vocab_word(Predictor *p, int min_len, char out[MAX_WORD_LEN])
{
    if (p->vocab_size < 5)
        return 0;
    for (int a = 0; a < 60; a++)
    {
        int idx = rand() % p->vocab_size;
        int wl = (int)strlen(p->vocab[idx]);
        if (wl >= min_len && wl < MAX_WORD_LEN - 1)
        {
            strncpy(out, p->vocab[idx], MAX_WORD_LEN - 1);
            out[MAX_WORD_LEN - 1] = 0;
            return 1;
        }
    }
    return 0;
}

static const char *FB_WORDS[] = {
    "language",
    "predict",
    "neural",
    "engine",
    "learning",
    "python",
    "machine",
    "science",
    "network",
    "pattern",
    "sequence",
    "grammar",
    "alphabet",
    "compute",
    "process",
    "program",
    "digital",
    "system",
    "analysis",
    "training",
    "dataset",
    "bigram",
    "context",
    "feature",
    "encoder",
    "decoder",
    "tensor",
    "matrix",
    "vector",
    "corpus",
    "token",
    "weight",
    "layer",
    "model",
    "output",
    "input",
    "class",
    "function",
    "variable",
    "boolean",
    "integer",
    "pointer",
    "memory",
};
static int FB_N = 40;

static void get_word(Predictor *p, int min_len, char out[MAX_WORD_LEN])
{
    if (!pick_vocab_word(p, min_len, out))
    {
        strncpy(out, FB_WORDS[rand() % FB_N], MAX_WORD_LEN - 1);
        out[MAX_WORD_LEN - 1] = 0;
    }
}

/* Check if a word exists in spell model OR fallback list */
static int word_is_valid(SpellCorrector *sc, Predictor *p, const char *w)
{
    /* 1. Check spell dictionary (hash) */
    if (sc && sc->dict_size > 0)
    {
        /* hash computed internally by spell_correct */
        /* We can't access sht directly here — use spell_correct with dist=0 */
        char qc[MAX_WORD_LEN];
        strncpy(qc, w, MAX_WORD_LEN - 1);
        qc[MAX_WORD_LEN - 1] = 0;
        SpellCandidate res[SPELL_TOP_K];
        int n = spell_correct(sc, qc, res);
        if (n > 0 && res[0].dist == 0)
            return 1;
        return 0;
    }
    /* 2. Check vocab flat array */
    if (p && p->vocab_size > 0)
    {
        for (int i = 0; i < p->vocab_size; i++)
            if (strcmp(p->vocab[i], w) == 0)
                return 1;
    }
    /* 3. Fallback list */
    for (int i = 0; i < FB_N; i++)
        if (strcmp(FB_WORDS[i], w) == 0)
            return 1;
    return 0;
}

/* Animated 3-2-1 Go */
static void countdown(void)
{
    const char *n[] = {"  3...", "  2...", "  1...", "  GO! 🚀"};
    const char *c[] = {WH, YL, GR, PU};
    for (int i = 0; i < 4; i++)
    {
        printf("\r%s%s%s   ", c[i], n[i], RS);
        fflush(stdout);
        if (i < 3)
        {
            Beep(500 + i * 200, 160);
            Sleep(700);
        }
        else
        {
            Beep(1200, 200);
            Sleep(300);
        }
    }
    printf("\n");
}

/* Timed input — reads word into buf, returns seconds taken.
   On platforms without select() (e.g. plain Windows console)
   we cannot do true non-blocking input, so we display a
   "START TYPING NOW" prompt, note the time before and after
   scanf, and if it exceeds limit we count as time-out.       */
static double timed_input(char *buf, int maxlen, int limit_sec)
{
    (void)maxlen;
    time_t t0 = time(NULL);
    if (scanf("%49s", buf) != 1)
        buf[0] = 0;
    clear_input_buffer();
    to_lowercase(buf);
    double spent = elapsed(t0);
    (void)limit_sec; /* limit enforcement done by caller */
    return spent;
}

/* Hint: reveal first N letters of word, mask rest with '_' */
static void show_hint(const char *bc, const char *word, int reveal, int att)
{
    int wl = (int)strlen(word);
    char hint[MAX_WORD_LEN * 3] = "";
    for (int i = 0; i < wl; i++)
    {
        if (i < reveal)
        {
            char c2[4];
            snprintf(c2, sizeof(c2), "%c", word[i]);
            strncat(hint, c2, sizeof(hint) - strlen(hint) - 1);
        }
        else
        {
            strncat(hint, "_", sizeof(hint) - strlen(hint) - 1);
        }
        strncat(hint, " ", sizeof(hint) - strlen(hint) - 1);
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "  💡 Hint #%d: " GR "%s" RS, att, hint);
    box_rowc(bc, buf);
}

/* Level selector — shows instructions + difficulty picker */
static int select_level(const char *game_name, const char *game_desc,
                        const char *rules[], int nrules)
{
    print_banner();
    printf("\n");
    box_top(PU);
    box_centerc(PU, PU "📜  GAME INSTRUCTIONS" RS);
    box_mid(PU);
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "  🎮  " YL "%s" RS, game_name);
        box_rowc(PU, buf);
    }
    box_rowc(PU, "");
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "  %s", game_desc);
        box_row(PU, buf);
    }
    box_sep(PU);
    box_rowc(PU, "  " CY "Rules:" RS);
    for (int i = 0; i < nrules; i++)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "  %s%s" RS, WH, rules[i]);
        box_rowc(PU, buf);
    }
    box_bot(PU);

    /* Difficulty menu */
    printf("\n");
    box_top(YL);
    box_centerc(YL, YL "⚙️   SELECT DIFFICULTY" RS);
    box_mid(YL);
    for (int i = 0; i < 3; i++)
    {
        char buf[256];
        const Level *lv = &LEVELS[i];
        snprintf(buf, sizeof(buf),
                 "  %s%d.  %-8s%s  " DM "%d rounds" RS
                 "  min-word: %d chars"
                 "  timer: %s%s%s"
                 "  attempts: %d",
                 lv->color, i + 1, lv->name, RS,
                 lv->rounds,
                 lv->min_word_len,
                 lv->time_limit > 0 ? YL : "", lv->time_limit > 0 ? "YES" : "OFF", RS,
                 lv->max_attempts);
        box_rowc(YL, buf);
        if (i < 2)
            box_sep(YL);
    }
    box_bot(YL);
    printf("  %s▶ Difficulty (1/2/3):%s ", PU, RS);
    fflush(stdout);
    int ch = 1;
    if (scanf("%d", &ch) != 1 || ch < 1 || ch > 3)
    {
        clear_input_buffer();
        ch = 1;
    }
    clear_input_buffer();
    return ch - 1; /* 0-based */
}

/* Shared results screen */
static void show_final(const char *game_name, int score, int max_score,
                       const char *grade, int lvl_idx)
{
    printf("\n");
    box_top(PU);
    box_centerc(PU, PU "🏆  GAME OVER" RS);
    box_mid(PU);
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "  🎮 Game:       " YL "%s" RS, game_name);
        box_rowc(PU, buf);
    }
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "  ⚙️   Difficulty: %s%s" RS,
                 LEVELS[lvl_idx].color, LEVELS[lvl_idx].name);
        box_rowc(PU, buf);
    }
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "  🏅 Score:      " GR "%d / %d" RS, score, max_score);
        box_rowc(PU, buf);
    }
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "  ⭐ Rating:     %s", grade);
        box_rowc(PU, buf);
    }
    box_bot(PU);
    beep_success();
}

// ============================================================
// GAME 1 — 🔀 WORD SCRAMBLE
// ============================================================

void game_word_scramble(Predictor *p)
{
    static const char *rules[] = {
        "A word is scrambled — rearrange the letters to find the original.",
        "You get multiple attempts depending on difficulty.",
        "Each wrong attempt reveals more of the word as a hint.",
        "Fewer attempts used = more points earned.",
        "Timer (Medium/Hard): answer before time runs out!",
    };
    int lvl = select_level(
        "🔀 WORD SCRAMBLE",
        "Unscramble the letters and reveal the hidden word.",
        rules, 5);
    const Level *L = &LEVELS[lvl];

    int total = 0, max_pts = L->rounds * L->pts_first;

    for (int round = 1; round <= L->rounds; round++)
    {
        print_banner();
        printf("\n");
        box_top(L->color);
        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "  🔀 WORD SCRAMBLE  ·  %s%s%s  ·  Round %s%d/%d%s  ·  Score: %s%d%s",
                     L->color, L->name, RS, YL, round, L->rounds, RS, GR, total, RS);
            box_rowc(L->color, buf);
        }
        box_sep(L->color);

        /* Get word */
        char word[MAX_WORD_LEN], scrambled[MAX_WORD_LEN];
        get_word(p, L->min_word_len, word);
        strncpy(scrambled, word, MAX_WORD_LEN - 1);
        scrambled[MAX_WORD_LEN - 1] = 0;
        int sa = 0;
        do
        {
            shuffle_str(scrambled, (int)strlen(scrambled));
            sa++;
        } while (strcmp(scrambled, word) == 0 && sa < 30);

        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "  🔀 Scrambled word:  " YL "%s" RS "   (" DM "%d letters" RS ")",
                     scrambled, (int)strlen(word));
            box_rowc(L->color, buf);
        }
        if (L->time_limit > 0)
            show_timer_row(L->color, L->time_limit);
        box_rowf(L->color,
                 "  Attempts allowed: " YL "%d" RS "   Points: " GR "%d/%d/%d" RS,
                 L->max_attempts, L->pts_first, L->pts_second, L->pts_third);
        box_bot(L->color);

        countdown();

        int pts = 0;
        char inp[MAX_WORD_LEN];
        int wlen = (int)strlen(word);

        for (int att = 1; att <= L->max_attempts && pts == 0; att++)
        {
            /* Show hint BEFORE each attempt (except attempt 1) */
            if (att > 1)
            {
                printf("\n");
                box_top(DM);
                /* Reveal more letters each time */
                int reveal = att; /* att=2→2 letters, att=3→3 letters */
                show_hint(DM, word, reveal, att - 1);
                box_rowc(DM, "  💡 Unscramble: " YL);
                /* Also show remaining letters as a sorted hint */
                char sorted[MAX_WORD_LEN];
                strncpy(sorted, word, MAX_WORD_LEN - 1);
                sorted[MAX_WORD_LEN - 1] = 0;
                /* bubble sort */
                int sl = (int)strlen(sorted);
                for (int a = 0; a < sl - 1; a++)
                    for (int b = 0; b < sl - a - 1; b++)
                        if (sorted[b] > sorted[b + 1])
                        {
                            char t = sorted[b];
                            sorted[b] = sorted[b + 1];
                            sorted[b + 1] = t;
                        }
                {
                    char buf[128];
                    snprintf(buf, sizeof(buf), "  📝 Letters available: " CY "%s" RS, sorted);
                    box_rowc(DM, buf);
                }
                box_bot(DM);
            }

            printf("  %s⌨️  Attempt %d/%d:%s ", L->color, att, L->max_attempts, RS);
            fflush(stdout);

            double spent = timed_input(inp, sizeof(inp), L->time_limit);
            (void)wlen;

            /* Timer check */
            if (L->time_limit > 0 && spent > (double)L->time_limit)
            {
                printf("\n");
                box_top(RD);
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  ⏰ " RD "TIME'S UP!" RS "  (%.1f s / %d s limit)  The word was: " YL "%s" RS,
                             spent, L->time_limit, word);
                    box_rowc(RD, b);
                }
                box_bot(RD);
                beep_error();
                pts = 0;
                break;
            }

            if (strcmp(inp, word) == 0)
            {
                int earned = (att == 1) ? L->pts_first : (att == 2) ? L->pts_second
                                                                    : L->pts_third;
                printf("\n");
                box_top(GR);
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  🎉 " GR "CORRECT!" RS "  Attempt %d  ·  +" GR "%d pts" RS
                             "  ·  Time: " CY "%.1fs" RS "  ·  Word: " YL "%s" RS,
                             att, earned, spent, word);
                    box_rowc(GR, b);
                }
                box_bot(GR);
                beep_success();
                pts = earned;
            }
            else
            {
                if (att < L->max_attempts)
                {
                    printf("  %s✗ Wrong!" RS, RD);
                    beep_error();
                    Sleep(300);
                    printf("  %sHint incoming on next attempt...%s\n", DM, RS);
                }
                else
                {
                    printf("\n");
                    box_top(RD);
                    {
                        char b[256];
                        snprintf(b, sizeof(b),
                                 "  ❌ " RD "Out of attempts!" RS "  The word was: " YL "%s" RS,
                                 word);
                        box_rowc(RD, b);
                    }
                    box_bot(RD);
                    beep_error();
                }
            }
        }
        total += pts;
        Sleep(800);
    }

    const char *grade =
        total >= max_pts ? GR "🌟 PERFECT!" RS : total >= max_pts * 2 / 3 ? YL "🔥 EXCELLENT" RS
                                             : total >= max_pts / 2       ? CY "✅ GOOD" RS
                                             : total >= max_pts / 4       ? WH "📈 KEEP TRYING" RS
                                                                          : DM "💪 PRACTICE!" RS;
    char grade_str[256];
    snprintf(grade_str, sizeof(grade_str), "%s", grade);
    show_final("Word Scramble", total, max_pts, grade_str, lvl);
    veryx_quip_for(0);
    wait_for_enter();
}

// ============================================================
// GAME 2 — 🔮 PREDICT-THE-NEXT
// ============================================================

void game_predict_next(Predictor *p)
{
    static const char *rules[] = {
        "VERYX shows you a word from its training data.",
        "Guess what word the model predicts comes NEXT.",
        "Top-1 exact match = full points.",
        "Anywhere in top-3 = partial points.",
        "Your vocabulary is the clue — think creatively!",
    };
    int lvl = select_level(
        "🔮 PREDICT-THE-NEXT",
        "Read VERYX's mind — predict what it thinks comes next.",
        rules, 5);
    const Level *L = &LEVELS[lvl];

    if (p->bigram_count < 20)
    {
        veryx_error("Need a bit more writing data — train the model first!");
        wait_for_enter();
        return;
    }

    int total = 0, max_pts = L->rounds * L->pts_first;

    for (int round = 1; round <= L->rounds; round++)
    {
        print_banner();

        /* Pick seed with predictions */
        char seed[MAX_WORD_LEN];
        char top_pred[TOP_K][MAX_WORD_LEN];
        Prediction preds[TOP_K];
        int np = 0, sa = 0;
        do
        {
            get_word(p, L->min_word_len, seed);
            np = predict_next_words(p, "", seed, preds);
            sa++;
        } while (np == 0 && sa < 40);

        if (np == 0)
        {
            printf("\n  %s⚠️  No prediction found — skipping round%s\n", YL, RS);
            Sleep(900);
            continue;
        }
        for (int i = 0; i < np && i < TOP_K; i++)
        {
            strncpy(top_pred[i], preds[i].word, MAX_WORD_LEN - 1);
            top_pred[i][MAX_WORD_LEN - 1] = 0;
        }

        printf("\n");
        box_top(L->color);
        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "  🔮 PREDICT-THE-NEXT  ·  %s%s%s  ·  Round %s%d/%d%s  ·  Score: %s%d%s",
                     L->color, L->name, RS, YL, round, L->rounds, RS, GR, total, RS);
            box_rowc(L->color, buf);
        }
        box_sep(L->color);
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  🔮 Seed word:  " YL "%s" RS "   →  What does VERYX predict next?", seed);
            box_rowc(L->color, buf);
        }
        /* Hard mode hint: show first letter of answer */
        if (lvl == 0)
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  💡 Hint: the next word starts with '" GR "%c" RS "'", top_pred[0][0]);
            box_rowc(L->color, buf);
        }
        if (L->time_limit > 0)
            show_timer_row(L->color, L->time_limit);
        box_bot(L->color);

        countdown();
        printf("  %s🔮 Your guess:%s ", CY, RS);
        fflush(stdout);

        char inp[MAX_WORD_LEN];
        double spent = timed_input(inp, sizeof(inp), L->time_limit);

        /* Timer check */
        if (L->time_limit > 0 && spent > (double)L->time_limit)
        {
            printf("\n");
            box_top(RD);
            {
                char b[256];
                snprintf(b, sizeof(b),
                         "  ⏰ " RD "TIME'S UP!" RS "  VERYX predicted: " YL "%s" RS, top_pred[0]);
                box_rowc(RD, b);
            }
            box_bot(RD);
            beep_error();
            Sleep(700);
            continue;
        }

        int pts = 0;
        if (strcmp(inp, top_pred[0]) == 0)
            pts = L->pts_first;
        else
            for (int i = 1; i < np; i++)
                if (strcmp(inp, top_pred[i]) == 0)
                {
                    pts = L->pts_third > 0 ? L->pts_third : 1;
                    break;
                }

        printf("\n");
        if (pts == L->pts_first)
        {
            box_top(GR);
            box_rowc(GR, "  🎉 " GR "PERFECT MATCH!" RS "  You read VERYX's mind!");
            {
                char b[256];
                snprintf(b, sizeof(b), "  Predicted: " YL "%s" RS "  ·  Time: " CY "%.1fs" RS "  ·  +" GR "%d pts" RS,
                         top_pred[0], spent, pts);
                box_rowc(GR, b);
            }
            box_bot(GR);
            beep_success();
        }
        else if (pts > 0)
        {
            box_top(YL);
            box_rowc(YL, "  🥈 " YL "In the top-3!" RS "  Close enough!");
            {
                char b[256];
                snprintf(b, sizeof(b), "  Top-1 was: " GR "%s" RS "  ·  +" YL "%d pt" RS, top_pred[0], pts);
                box_rowc(YL, b);
            }
            box_bot(YL);
            Beep(900, 120);
            Sleep(60);
            Beep(1100, 120);
        }
        else
        {
            box_top(RD);
            {
                char b[256];
                char t3[64] = "";
                for (int i = 0; i < np && i < 3; i++)
                {
                    if (i)
                        strncat(t3, ", ", sizeof(t3) - strlen(t3) - 1);
                    strncat(t3, top_pred[i], sizeof(t3) - strlen(t3) - 1);
                }
                snprintf(b, sizeof(b), "  ❌ VERYX predicted: " YL "%s" RS "  ·  You guessed: " RD "%s" RS, t3, inp);
                box_rowc(RD, b);
            }
            box_bot(RD);
            beep_error();
        }
        total += pts;
        Sleep(800);
    }

    const char *grade =
        total >= max_pts ? GR "🧠 MIND READER" RS : total >= max_pts * 2 / 3 ? YL "🔮 PSYCHIC" RS
                                                : total >= max_pts / 2       ? CY "🎯 SHARP" RS
                                                : total >= max_pts / 4       ? WH "📖 STUDENT" RS
                                                                             : DM "🌱 BEGINNER" RS;
    char gs[256];
    snprintf(gs, sizeof(gs), "%s", grade);
    show_final("Predict-the-Next", total, max_pts, gs, lvl);
    veryx_quip_for(5);
    wait_for_enter();
}

// ============================================================
// GAME 3 — ⚔️  WORD DUEL
// ============================================================
/*
 * VERYX plays a word. Player must type a LONGER AND VALID word.
 * Validity = checked against spell model or vocab.
 * Time limit applies on Medium/Hard.
 * Hints: if wrong, shows what letter range is needed.
 */

void game_word_duel(Predictor *p, SpellCorrector *sc)
{
    static const char *rules[] = {
        "VERYX plays a word. You must type a LONGER valid word.",
        "Your word MUST be a real word (spell-checked against the model).",
        "Longer + valid = full points.",
        "Longer but not in dictionary = partial points (2 pts).",
        "Hints are shown when your word fails the length or spelling check.",
        "Timer enforced on Medium and Hard difficulty!",
    };
    int lvl = select_level(
        "⚔️  WORD DUEL",
        "Beat VERYX with a longer, real word — and beat the clock!",
        rules, 6);
    const Level *L = &LEVELS[lvl];
    int total = 0, max_pts = L->rounds * L->pts_first;

    for (int round = 1; round <= L->rounds; round++)
    {
        print_banner();

        char vword[MAX_WORD_LEN];
        get_word(p, L->min_word_len, vword);
        int vlen = (int)strlen(vword);

        printf("\n");
        box_top(L->color);
        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "  ⚔️  WORD DUEL  ·  %s%s%s  ·  Round %s%d/%d%s  ·  Score: %s%d%s",
                     L->color, L->name, RS, YL, round, L->rounds, RS, GR, total, RS);
            box_rowc(L->color, buf);
        }
        box_sep(L->color);
        {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "  ⚔️   VERYX plays: " YL "%s" RS "  (length: " CY "%d" RS
                     ")   Beat it with a LONGER real word!",
                     vword, vlen);
            box_rowc(L->color, buf);
        }
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "  📏 Your word must be at least " GR "%d" RS " letters long.",
                     vlen + 1);
            box_rowc(L->color, buf);
        }
        if (L->time_limit > 0)
            show_timer_row(L->color, L->time_limit);
        box_bot(L->color);

        countdown();

        /* Allow up to 2 attempts for Easy, 1 for others */
        int max_att = (lvl == 0) ? 2 : 1;
        int pts = 0;
        char inp[MAX_WORD_LEN];
        double spent = 0;

        for (int att = 1; att <= max_att && pts == 0; att++)
        {
            printf("  %s⚔️  Your word (attempt %d/%d):%s ",
                   L->color, att, max_att, RS);
            fflush(stdout);
            spent = timed_input(inp, sizeof(inp), L->time_limit);

            /* Timer check */
            if (L->time_limit > 0 && spent > (double)L->time_limit)
            {
                printf("\n");
                box_top(RD);
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  ⏰ " RD "TIME'S UP!" RS "  (%.1f s / %d s)  VERYX wins this round!",
                             spent, L->time_limit);
                    box_rowc(RD, b);
                }
                box_bot(RD);
                beep_error();
                pts = 0;
                break;
            }

            int ilen = (int)strlen(inp);
            if (ilen <= 0)
            {
                beep_error();
                continue;
            }

            /* Spell check */
            int valid = word_is_valid(sc, p, inp);

            /* Evaluate */
            if (ilen > vlen && valid)
            {
                pts = L->pts_first;
                printf("\n");
                box_top(GR);
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  🎉 " GR "PERFECT!" RS
                             "  " GR "%s" RS " (%d) > " RD "%s" RS " (%d)"
                             "  ✅ valid word  ·  +" GR "%d pts" RS "  ·  %.1fs",
                             inp, ilen, vword, vlen, pts, spent);
                    box_rowc(GR, b);
                }
                box_bot(GR);
                beep_success();
            }
            else if (ilen > vlen && !valid)
            {
                pts = L->pts_second; /* longer but not in dictionary */
                printf("\n");
                box_top(YL);
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  🟡 " YL "Longer, but '" RD "%s" YL "' not found in dictionary." RS
                             "  +" YL "%d pts" RS " (no spell bonus)",
                             inp, pts);
                    box_rowc(YL, b);
                }
                box_bot(YL);
                Beep(700, 120);
            }
            else
            {
                /* Wrong — give hints */
                printf("\n");
                box_top(RD);
                if (ilen <= vlen)
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  ❌ " RD "Too short!" RS
                             "  " RD "%s" RS " = %d letters ·  Need at least " YL "%d" RS " letters",
                             inp, ilen, vlen + 1);
                    box_rowc(RD, b);
                    if (att < max_att)
                    {
                        char hint[256];
                        snprintf(hint, sizeof(hint),
                                 "  💡 Hint: Try a word with " YL "%d+" RS " letters."
                                 " Example range: " CY "a...z" RS " combinations of length %d",
                                 vlen + 1, vlen + 1);
                        box_rowc(RD, hint);
                    }
                }
                else if (!valid)
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  ❌ " RD "Not a recognised word!" RS
                             "  '" RD "%s" RS "' not in dictionary.",
                             inp);
                    box_rowc(RD, b);
                    if (att < max_att)
                    {
                        char hint[256];
                        snprintf(hint, sizeof(hint),
                                 "  💡 Hint: Train more data, or try a common word"
                                 " with " YL "%d+" RS " letters.",
                                 vlen + 1);
                        box_rowc(RD, hint);
                    }
                }
                box_bot(RD);
                beep_error();
                if (att == max_att)
                {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "\n  %s⚔️  VERYX wins this round! VERYX word: %s%s%s (%d)%s",
                             RD, YL, vword, RS, vlen, RS);
                    printf("%s\n", b);
                }
            }
        }
        total += pts;
        Sleep(700);
    }

    const char *grade =
        total >= max_pts ? GR "⚔️  WORD CHAMPION" RS : total >= max_pts * 2 / 3 ? YL "🏹 SHARP DUELIST" RS
                                                  : total >= max_pts / 2       ? CY "🛡️  SOLID FIGHTER" RS
                                                  : total >= max_pts / 4       ? WH "🗡️  APPRENTICE" RS
                                                                               : DM "🌱 KEEP PRACTISING" RS;
    char gs[256];
    snprintf(gs, sizeof(gs), "%s", grade);
    show_final("Word Duel", total, max_pts, gs, lvl);
    veryx_quip_for(0);
    wait_for_enter();
}

// ============================================================
// GAME ARCADE MENU
// ============================================================

void game_arcade(Predictor *p, SpellCorrector *sc)
{
    int choice;
    do
    {
        print_banner();
        printf("\n");
        box_top(PU);
        box_centerc(PU, YL "🕹️   VERYX GAME ARCADE" RS "  " PU "Play · Learn · Compete" RS);
        box_sep(PU);

        /* Model status hint */
        if (p->vocab_size > 0)
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  %s✅  Model loaded: %d words · %d bigrams%s",
                     GR, p->vocab_size, p->bigram_count, RS);
            box_rowc(PU, buf);
        }
        else
        {
            box_rowc(PU, "  " YL "⚠️   No model — games use built-in word list" RS);
        }
        if (sc->dict_size > 0)
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  %s✅  Spell model: %d words (used for Word Duel)%s",
                     CY, sc->dict_size, RS);
            box_rowc(PU, buf);
        }

        box_sep(PU);
        box_rowc(PU, "  " YL "1." RS "  🔀  " GR "Word Scramble" RS
                     "       — Unscramble scrambled letters");
        box_rowc(PU, "  " YL "2." RS "  🔮  " CY "Predict-the-Next" RS
                     "   — Guess VERYX's next-word prediction");
        box_rowc(PU, "  " YL "3." RS "  ⚔️   " RD "Word Duel" RS
                     "          — Beat VERYX with a longer valid word");
        box_sep(PU);
        box_rowc(PU, "  " RD "4." RS "  🔙  Back to dashboard");
        box_bot(PU);

        printf("  %s▶ Choice:%s ", PU, RS);
        fflush(stdout);
        if (scanf("%d", &choice) != 1)
        {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();

        switch (choice)
        {
        case 1:
            game_word_scramble(p);
            break;
        case 2:
            game_predict_next(p);
            break;
        case 3:
            game_word_duel(p, sc);
            break;
        case 4:
            break;
        default:
            veryx_error("Choose 1-4");
            wait_for_enter();
        }
    } while (choice != 4);
}

// ============================================================
// SENTENCE BUILDER
// ============================================================
void build_sentence(Predictor *p)
{
    print_banner();
    printf("\n");
    box_top(YL);
    box_rowc(YL, "  " YL "⚡ VERYX" RS "  ✍️  INTERACTIVE SENTENCE BUILDER");
    box_mid(YL);
    box_rowc(YL, "  " WH "Type words one at a time." RS);
    box_row(YL, "  Type '.'  to end  |  'quit'  to cancel");
    box_bot(YL);
    if (!p->vocab_size)
    {
        veryx_error("Model not trained! Ask an admin to train it.");
        wait_for_enter();
        return;
    }
    char **sent = (char **)malloc(MAX_SENTENCE_LENGTH * sizeof(char *));
    int wc = 0;
    char inp[MAX_WORD_LEN];
    if (!sent)
    {
        veryx_error("Memory error");
        wait_for_enter();
        return;
    }
    while (wc < MAX_SENTENCE_LENGTH)
    {
        printf("\n  %sSentence:%s ", YL, RS);
        if (!wc)
            printf("%s[start typing...]%s", DM, RS);
        else
            for (int i = 0; i < wc; i++)
                printf("%s%s%s ", GR, sent[i], RS);
        printf("\n  %s⚡ VERYX%s  %s✍️  Next word:%s ", YL, RS, GR, RS);
        fflush(stdout);
        if (scanf("%49s", inp) != 1)
        {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();
        if (!strcmp(inp, "quit"))
            break;
        if (!strcmp(inp, "."))
        {
            printf("\n  %s✅ Sentence done.%s\n", GR, RS);
            beep_success();
            break;
        }
        sent[wc] = (char *)malloc(strlen(inp) + 1);
        if (!sent[wc])
            break;
        strcpy(sent[wc], inp);
        wc++;
        if (wc == 3)
            veryx_maybe_remark(1); /* always at word 3 */
        else if (wc > 3)
            veryx_maybe_remark(6); /* 1-in-6 after that */
        printf("\n  %s⚡ VERYX%s  consulting model...\n", YL, RS);
        neural_pulse(5);
        beep_thinking();
        Prediction res[TOP_K];
        int nr;
        if (wc >= 2)
            nr = predict_next_words(p, sent[wc - 2], sent[wc - 1], res);
        else if (wc == 1)
            nr = predict_next_words(p, "", sent[0], res);
        else
            nr = 0;
        if (nr > 0)
            print_predictions(res, nr);
        else
            printf("  %s⚡ VERYX%s  No predictions yet.\n", YL, RS);
    }
    printf("\n");
    box_top(GR);
    box_rowc(GR, "  " YL "📝 FINAL SENTENCE" RS);
    box_mid(GR);
    char full[MAX_LINE * 2] = "  ";
    for (int i = 0; i < wc; i++)
    {
        strncat(full, sent[i], sizeof(full) - strlen(full) - 2);
        if (i < wc - 1)
            strncat(full, " ", sizeof(full) - strlen(full) - 1);
    }
    /* wrap at TERM_W */
    int flen = (int)strlen(full);
    for (int off = 0; off < flen; off += TERM_W - 2)
    {
        char chunk[TERM_W + 1];
        snprintf(chunk, sizeof(chunk), "  %.*s", TERM_W - 2, full + off);
        box_row(GR, chunk);
    }
    box_bot(GR);
    beep_success();
    {
        for (int i = 0; i < wc; i++)
            free(sent[i]);
    }
    free(sent);
    veryx_quip_for(2);
    wait_for_enter();
}

void auto_generate_sentence(Predictor *p)
{
    print_banner();
    printf("\n");
    box_top(YL);
    box_rowc(YL, "  " YL "⚡ VERYX" RS "  🔄 AUTO-GENERATE SENTENCE");
    box_bot(YL);
    if (!p->vocab_size)
    {
        veryx_error("Model not trained! Ask an admin.");
        wait_for_enter();
        return;
    }
    char sw[MAX_WORD_LEN];
    printf("\n  %s🔤 Starting word:%s ", PU, RS);
    fflush(stdout);
    if (scanf("%49s", sw) != 1)
    {
        clear_input_buffer();
        wait_for_enter();
        return;
    }
    clear_input_buffer();
    to_lowercase(sw);
    printf("\n  %s1%s. 📝 Natural ending    %s2%s. 🔢 Fixed word count\n  %s▶%s Choice: ", GR, RS, GR, RS, PU, RS);
    fflush(stdout);
    int mode;
    if (scanf("%d", &mode) != 1 || (mode != 1 && mode != 2))
    {
        clear_input_buffer();
        veryx_error("Invalid");
        wait_for_enter();
        return;
    }
    clear_input_buffer();
    int wl = 0;
    if (mode == 2)
    {
        printf("  %s🔢 Words (5-50):%s ", PU, RS);
        fflush(stdout);
        if (scanf("%d", &wl) != 1)
        {
            clear_input_buffer();
            wl = 15;
        }
        clear_input_buffer();
        if (wl < 5)
            wl = 5;
        if (wl > 50)
            wl = 50;
    }
    veryx_thinking("Crafting your sentence from learned patterns");
    scan_animation("GENERATING");
    neural_pulse(6);
    beep_generate();
    char *s = generate_ngram_sentence(p, sw, wl);
    printf("\n");
    box_top(PU);
    box_rowc(PU, "  " YL "✨ VERYX" RS "  GENERATED SENTENCE");
    box_mid(PU);
    int sl = (int)strlen(s);
    for (int off = 0; off < sl; off += TERM_W - 2)
    {
        char chunk[TERM_W + 1];
        snprintf(chunk, sizeof(chunk), "  %.*s", TERM_W - 2, s + off);
        box_row(PU, chunk);
    }
    box_mid(PU);
    box_rowf(PU, "  Length: %d chars   Mode: %s", (int)strlen(s), (mode == 1 ? "Natural" : "Fixed"));
    box_bot(PU);
    beep_success();
    veryx_quip_for(5);
    wait_for_enter();
}

void style_mimic_demo(Predictor *p)
{
    print_banner();
    if (!p->topic_count)
    {
        veryx_error("No topics. Ask admin to train the model.");
        wait_for_enter();
        return;
    }
    printf("\n");
    box_top(PU);
    box_centerc(PU, YL "⚡ VERYX" RS "  " PU "🎭  STYLE MIMIC  ─  Choose a Topic" RS);
    box_sep(PU);
    box_rowc(PU, "  " DM "  #    Topic              Related Keywords" RS);
    box_sep(PU);
    for (int i = 0; i < p->topic_count; i++)
    {
        /* Build coloured keyword tags */
        char tags[256] = "";
        const char *kcolors[] = {GR, CY, YL};
        int kc = 0;
        for (int k = 0; k < STYLE_WORDS && kc < 3; k++)
        {
            if (*p->keywords[i][k] && strcmp(p->keywords[i][k], p->topics[i]))
            {
                char tag[64];
                snprintf(tag, sizeof(tag), "%s[%s]%s ", kcolors[kc % 3], p->keywords[i][k], RS);
                strncat(tags, tag, sizeof(tags) - strlen(tags) - 1);
                kc++;
            }
        }
        char row[512];
        const char *nc = (i % 3 == 0) ? YL : (i % 3 == 1) ? CY
                                                          : WH;
        snprintf(row, sizeof(row), "  %s%2d.%s  %s%-16s%s  %s",
                 nc, i + 1, RS, PU, p->topics[i], RS, tags);
        box_rowc(PU, row);
        if (i < p->topic_count - 1)
            box_sep(PU);
    }
    box_bot(PU);
    int tc;
    printf("\n  %s🎭 Topic (1-%d):%s ", PU, p->topic_count, RS);
    fflush(stdout);
    if (scanf("%d", &tc) != 1)
    {
        clear_input_buffer();
        wait_for_enter();
        return;
    }
    clear_input_buffer();
    if (tc < 1 || tc > p->topic_count)
    {
        veryx_error("Invalid topic");
        wait_for_enter();
        return;
    }
    printf("\n  %s1%s. 📝 Natural  %s2%s. 🔢 Fixed\n  %s▶%s Choice: ", GR, RS, GR, RS, PU, RS);
    fflush(stdout);
    int mode;
    if (scanf("%d", &mode) != 1 || (mode != 1 && mode != 2))
    {
        clear_input_buffer();
        mode = 1;
    }
    clear_input_buffer();
    int wl = 0;
    if (mode == 2)
    {
        printf("  %s🔢 Words(5-50):%s ", PU, RS);
        fflush(stdout);
        if (scanf("%d", &wl) != 1)
        {
            clear_input_buffer();
            wl = 15;
        }
        clear_input_buffer();
        if (wl < 5)
            wl = 5;
        if (wl > 50)
            wl = 50;
    }
    veryx_thinking("Composing style-matched sentence");
    neural_pulse(8);
    beep_generate();
    char *s = generate_style_sentence(p, p->topics[tc - 1], wl);
    /* ── Display the generated sentence beautifully ──────────── */
    printf("\n");
    box_top(PU);
    {
        char hdr[200];
        snprintf(hdr, sizeof(hdr), " 🎭  " PU "STYLE MIMIC" RS "  ─  Topic: " YL "%s" RS, p->topics[tc - 1]);
        box_rowc(PU, hdr);
    }
    box_mid(PU);
    /* Render sentence word by word with alternating colors */
    {
        /* Tokenise s into words */
        char scopy[MAX_LINE * 2];
        strncpy(scopy, s, sizeof(scopy) - 1);
        scopy[sizeof(scopy) - 1] = 0;
        const char *wcolors[] = {GR, CY, YL, WH, PU};
        int nwc = 5;
        /* Build a coloured display line that wraps at TERM_W-4 printable chars */
        char linebuf[1024] = "  ";
        int linevis = 2, widx = 0;
        char *tok2 = strtok(scopy, " ");
        while (tok2)
        {
            int wlen = (int)strlen(tok2);
            /* Will this word + space fit? */
            if (linevis + wlen + 1 > TERM_W - 4 && linevis > 2)
            {
                /* flush current line */
                box_rowc(PU, linebuf);
                strcpy(linebuf, "  ");
                linevis = 2;
            }
            /* append coloured word */
            const char *wc3 = wcolors[widx % nwc];
            char wordbuf[256];
            snprintf(wordbuf, sizeof(wordbuf), "%s%s%s ", wc3, tok2, RS);
            strncat(linebuf, wordbuf, sizeof(linebuf) - strlen(linebuf) - 1);
            linevis += wlen + 1;
            widx++;
            tok2 = strtok(NULL, " ");
        }
        if (linevis > 2)
            box_rowc(PU, linebuf);
    }
    box_mid(PU);
    /* Word count + topic info */
    {
        /* Count words in s */
        int wcount = 0;
        const char *p2 = s;
        while (*p2)
        {
            while (*p2 == ' ')
                p2++;
            if (*p2)
            {
                wcount++;
                while (*p2 && *p2 != ' ')
                    p2++;
            }
        }
        char info[200];
        snprintf(info, sizeof(info),
                 "  📊  " DM "Words: " GR "%d" RS "   Mode: " CY "%s" RS "   Topic: " YL "%s" RS,
                 wcount, (mode == 1 ? "Natural" : "Fixed"), p->topics[tc - 1]);
        box_rowc(PU, info);
    }
    box_bot(PU);
    beep_success();
    veryx_quip_for(5);
    wait_for_enter();
}

// ============================================================
// ADMIN USER MANAGEMENT
// ============================================================
void admin_list_users(UserDB *db)
{
    print_banner();
    printf("\n");
    box_top(YL);
    box_rowf(YL, "  👥 REGISTERED USERS  ─  " YL "%d" RS " total", db->count);
    box_mid(YL);
    box_rowc(YL, "  " DM "##   Username              Role    Created" RS);
    box_sep(YL);
    for (int i = 0; i < db->count; i++)
    {
        char line[TERM_W + 1];
        const char *rl = (db->users[i].role == ROLE_ADMIN) ? "Admin " : "User  ";
        snprintf(line, sizeof(line), "  %02d   %-20.20s  %s  %.10s", i + 1, db->users[i].username, rl, db->users[i].created);
        box_row(YL, line);
        if (i < db->count - 1)
            box_sep(YL);
    }
    box_bot(YL);
    wait_for_enter();
}

void admin_delete_user(UserDB *db, const char *cur)
{
    print_banner();
    printf("\n");
    box_top(RD);
    box_rowc(RD, "  " RD "❌ DELETE USER" RS);
    box_bot(RD);
    printf("  %s❌ Username to delete:%s ", RD, RS);
    fflush(stdout);
    char un[MAX_USERNAME] = "";
    if (fgets(un, sizeof(un), stdin) == NULL)
        return;
    un[strcspn(un, "\n")] = 0;
    if (!strcmp(un, cur))
    {
        veryx_error("Cannot delete your own account");
        wait_for_enter();
        return;
    }
    int idx = userdb_find(db, un);
    if (idx < 0)
    {
        veryx_error("User not found");
        wait_for_enter();
        return;
    }
    {
        for (int i = idx; i < db->count - 1; i++)
            db->users[i] = db->users[i + 1];
    }
    db->count--;
    save_userdb(db);
    veryx_success("User deleted successfully");
    wait_for_enter();
}

// ============================================================
// ADMIN MENU
// ============================================================
void admin_menu(Predictor *p, SpellCorrector *sc, UserDB *db, Session *sess)
{
    int choice;
    char fn[MAX_LINE];
    do
    {
        print_banner();
        veryx_maybe_remark(5); /* 20% chance per menu visit */
        printf("\n");
        box_top(YL);
        box_rowf(YL, "  🛡️  ADMIN PANEL  ─  " YL "%s" RS, sess->username);
        box_mid(YL);
        box_rowc(YL, "  " CY "── 🧠 N-GRAM MODEL ─────────────────────" RS);
        box_rowc(YL, "  " GR "1." RS " 🚂 Train N-gram model on a data file");
        box_rowc(YL, "  " GR "2." RS " 📊 View N-gram statistics");
        box_rowc(YL, "  " RD "3." RS " 🗑️  Reset N-gram model");
        box_sep(YL);
        box_rowc(YL, "  " PU "── 🔤 SPELL MODEL ──────────────────────" RS);
        box_rowc(YL, "  " GR "4." RS " 📖 Train spell corrector on a file");
        box_rowc(YL, "  " GR "5." RS " 📊 View spell statistics");
        box_rowc(YL, "  " RD "6." RS " 🗑️  Reset spell model");
        box_sep(YL);
        box_rowc(YL, "  " YL "── 👥 USER MANAGEMENT ──────────────────" RS);
        box_rowc(YL, "  " GR "7." RS " 👤 View all registered users");
        box_rowc(YL, "  " RD "8." RS " ❌ Delete a user account");
        box_sep(YL);
        box_rowc(YL, "  " GR "9." RS " ✨ Use VERYX features");
        box_rowc(YL, "  " RD "0." RS " 🚪 Log out");
        box_bot(YL);
        printf("  %s▶ Choice:%s ", PU, RS);
        fflush(stdout);
        if (scanf("%d", &choice) != 1)
        {
            clear_input_buffer();
            veryx_error("Invalid input");
            wait_for_enter();
            continue;
        }
        clear_input_buffer();
        switch (choice)
        {
        case 1:
            print_banner();
            printf("\n");
            box_top(GR);
            box_rowc(GR, "  " GR "🚂 N-GRAM TRAINING MODE" RS);
            box_bot(GR);
            printf("  %s📁 Filename:%s ", CY, RS);
            fflush(stdout);
            if (fgets(fn, sizeof(fn), stdin))
            {
                fn[strcspn(fn, "\n")] = 0;
                if (*fn)
                    train_from_file(p, fn);
            }
            wait_for_enter();
            break;
        case 2:
            print_banner();
            print_stats(p);
            wait_for_enter();
            break;
        case 3:
        {
            print_banner();
            printf("  %s⚠️  Reset N-gram? (y/n):%s ", RD, RS);
            fflush(stdout);
            char c;
            if (scanf(" %c", &c) == 1 && (c == 'y' || c == 'Y'))
                reset_model(p);
            else
            {
                printf("  Cancelled.\n");
                beep_error();
            }
            clear_input_buffer();
            wait_for_enter();
            break;
        }
        case 4:
            print_banner();
            printf("\n");
            box_top(CY);
            box_rowc(CY, "  " CY "📖 SPELL TRAINING MODE" RS);
            box_bot(CY);
            printf("  %s📁 Filename:%s ", CY, RS);
            fflush(stdout);
            if (fgets(fn, sizeof(fn), stdin))
            {
                fn[strcspn(fn, "\n")] = 0;
                if (*fn)
                    spell_train_from_file(sc, fn);
            }
            wait_for_enter();
            break;
        case 5:
            print_banner();
            spell_print_stats(sc);
            wait_for_enter();
            break;
        case 6:
        {
            print_banner();
            printf("  %s⚠️  Reset spell? (y/n):%s ", RD, RS);
            fflush(stdout);
            char c;
            if (scanf(" %c", &c) == 1 && (c == 'y' || c == 'Y'))
                spell_reset(sc);
            else
            {
                printf("  Cancelled.\n");
                beep_error();
            }
            clear_input_buffer();
            wait_for_enter();
            break;
        }
        case 7:
            admin_list_users(db);
            break;
        case 8:
            admin_delete_user(db, sess->username);
            break;
        case 9:
            user_menu(p, sc, sess);
            break;
        case 0:
            veryx_info("Logging out...");
            beep_success();
            Sleep(700);
            sess->logged_in = 0;
            break;
        default:
            veryx_error("Invalid choice");
            wait_for_enter();
        }
    } while (choice != 0);
}

// ============================================================
// USER MENU
// ============================================================
void user_menu(Predictor *p, SpellCorrector *sc, Session *sess)
{
    int choice;
    do
    {
        print_banner();
        veryx_maybe_remark(4); /* 25% chance per menu visit */
        printf("\n");
        box_top(WH);
        box_rowf(WH, "  👤 USER DASHBOARD  ─  " GR "%s" RS, sess->username);
        box_mid(WH);
        box_rowc(WH, "  " GR "1." RS " ✍️  Build a sentence interactively");
        box_rowc(WH, "  " GR "2." RS " 🎭 Style Mimic — generate in topic style");
        box_rowc(WH, "  " GR "3." RS " 🔄 Auto-generate a sentence");
        box_rowc(WH, "  " GR "4." RS " 🔤 Spell Corrector — check your words");
        box_rowc(WH, "  " GR "5." RS " 📊 View model statistics");
        box_rowc(WH, "  " GR "6." RS " 🕹️  Game Arcade — Play & Learn!");
        box_sep(WH);
        if (sess->role == ROLE_USER)
            box_rowc(WH, "  " RD "7." RS " 🚪 Log out");
        else
            box_rowc(WH, "  " YL "7." RS " 🔙 Back to Admin Panel");
        box_bot(WH);
        if (!p->vocab_size)
            printf("  %s⚠️  N-gram model not trained yet — ask an admin%s\n", YL, RS);
        if (!sc->dict_size)
            printf("  %s⚠️  Spell model not trained yet — ask an admin%s\n", YL, RS);
        printf("  %s▶ Choice:%s ", PU, RS);
        fflush(stdout);
        if (scanf("%d", &choice) != 1)
        {
            clear_input_buffer();
            veryx_error("Invalid input");
            wait_for_enter();
            continue;
        }
        clear_input_buffer();
        switch (choice)
        {
        case 1:
            build_sentence(p);
            break;
        case 2:
            style_mimic_demo(p);
            break;
        case 3:
            auto_generate_sentence(p);
            break;
        case 4:
            spell_correct_interactive(sc);
            wait_for_enter();
            break;
        case 5:
            print_banner();
            print_stats(p);
            spell_print_stats(sc);
            if (rand() % 2 == 0)
                veryx_quip_for(6);
            wait_for_enter();
            break;
        case 6:
            game_arcade(p, sc);
            break;
        case 7:
            if (sess->role == ROLE_USER)
            {
                veryx_info("Logging out...");
                beep_success();
                Sleep(700);
                sess->logged_in = 0;
            }
            break;
        default:
            veryx_error("Invalid choice");
            wait_for_enter();
        }
    } while (choice != 7);
}

// ============================================================
// MAIN
// ============================================================
int main(void)
{
#ifdef _WIN32
    system("chcp 65001");
    system("title VERYX Neural Engine v10.0");
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif
    setbuf(stdout, NULL);
    srand((unsigned int)time(NULL));

    boot_sequence();
    ai_logs();
    loading_bar("Warming Up the VERYX Creative Engine");
    thinking_animation(2);
    alert("[SUCCESS] VERYX v10.0 ONLINE ⚡");
    glitch_text("  VERYX — Secure Predictive Intelligence");
    Sleep(200);
    type_text("  Loading secure interactive environment...");
    Sleep(300);

    Predictor predictor;
    SpellCorrector speller;
    UserDB userdb;
    Session session;

    init_predictor(&predictor);
    init_spell(&speller);
    init_userdb(&userdb);
    session.logged_in = 0;
    session.role = ROLE_USER;
    session.username[0] = 0;

    printf("\n  %s⚡ VERYX%s  loading saved data...\n", YL, RS);
    neural_pulse(6);
    auto_load(&predictor);
    spell_auto_load(&speller);
    load_userdb(&userdb);

    while (1)
    {
        auth_screen(&userdb, &session);
        if (session.role == ROLE_ADMIN)
            admin_menu(&predictor, &speller, &userdb, &session);
        else
            user_menu(&predictor, &speller, &session);
        session.logged_in = 0;
    }

    free_predictor(&predictor);
    free_spell(&speller);
    free_userdb(&userdb);
    return 0;
}
