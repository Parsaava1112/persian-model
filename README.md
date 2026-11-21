# Danory GPT v7 — مدل زبانی فارسی از ابتدا

نسخهٔ مستقل v7 با معماری تازه، توکنایزر جدید و آموزش از وزن‌های تصادفی ساخته می‌شود. **هیچ وزن، optimizer state یا checkpoint از v2 تا v6 بارگذاری نمی‌شود.** فایل‌های v6 نیز در این پروژه وجود ندارند.

## ۱) مشخصات معماری

| ویژگی | مقدار |
|---|---:|
| تعداد پارامترها | 23,452,032 (حدود 23.45 میلیون) |
| Vocabulary | 16,000 |
| Context length | 1,024 توکن |
| Transformer layers | 8 |
| Embedding dimension | 384 |
| Query heads | 6 |
| KV heads | 2 |
| FFN dimension | 1,536 |
| Attention | Causal GQA + RoPE |
| Feed-forward | SwiGLU |
| Normalization | RMSNorm |
| Embedding/output | وزن‌های مشترک (tied embeddings) |
| Optimizer | AdamW |
| LR schedule | Warmup + Cosine decay |

تعداد پارامترها با `python scripts/model_info.py` قابل محاسبه است. این مدل یک مدل کوچک پژوهشی است، نه مدلی در مقیاس تجاری. کیفیت نهایی به مقدار دادهٔ قابل‌استفاده، تعداد توکن‌های دیده‌شده، تعداد مراحل آموزش و ارزیابی بستگی دارد.

## ۲) داده‌ها و مجوزها

### پیش‌آموزش اصلی: `SLPL/naab`

- مخزن: https://huggingface.co/datasets/SLPL/naab
- پیکرهٔ فارسی پاک‌سازی‌شده و شامل متن‌های رسمی و محاوره‌ای است.
- کارت دیتاست حدود 130GB، نزدیک به 250 میلیون پاراگراف و حدود 15 میلیارد واژه را گزارش می‌کند.
- ساختار نمونه‌ها دارای فیلد `text` است و splitهای `train` و `test` دارد.
- کارت Hugging Face مجوز MIT را نشان می‌دهد؛ اما فایل `naab.py` در بخش citation عبارت CC BY-NC-SA 4.0 را هم ذکر می‌کند، در حالی که فیلد مجوز کد `mit` است. این ناهمخوانی باید پیش از انتشار یا استفادهٔ تجاری با نگه‌دارندگان دیتاست روشن شود. همچنین چون مجموعه از چند منبع گردآوری شده، شرایط منابع تشکیل‌دهنده و ملاحظات حریم خصوصی را بررسی کنید.

این پروژه کل 130GB را روی GitHub یا رایانه دانلود نمی‌کند. نکتهٔ فنی مهم این است که loader قدیمی NAAM در `naab.py` همهٔ فایل‌ها را به `dl_manager.download` می‌دهد؛ بنابراین `streaming=True` برای آن روش می‌تواند به دانلود تمام فایل‌ها منجر شود. در این پروژه فقط shardهای انتخاب‌شده با `hf_hub_download` دریافت می‌شوند. هر shard آموزشی حدود 1.07GB است؛ shard آزمون شمارهٔ 2 حدود 287MB است. تنظیم پیش‌فرض فقط train shard شمارهٔ 0 و test shard شمارهٔ 2 را می‌گیرد، سپس به‌ترتیب حداکثر 300,000 و 20,000 پاراگراف را استخراج می‌کند و فایل shardهای خام را حذف می‌کند. شمارهٔ shardها در Workflow قابل تغییر است. برای نمونه، `0,1` دو shard آموزشی را انتخاب می‌کند. برای جلوگیری از پرشدن دیسک runner، در هر اجرا حداکثر 4 shard آموزشی و یک shard آزمون مجاز است؛ سقف ردیف‌های آموزشی نیز یک میلیون است. این روش از دانلود تصادفی کل 130GB جلوگیری می‌کند و در عین حال محدودیت 14GB runner را رعایت می‌کند.

### مجموعهٔ تکمیلی احتمالی

`mshojaei77/PersianCorpus_merged` یک مجموعهٔ تجمیع‌شدهٔ دیگر با حدود 14.7 میلیون ردیف و حجم اعلام‌شدهٔ 24.6GB است و کارت آن MIT را نشان می‌دهد: https://huggingface.co/datasets/mshojaei77/PersianCorpus_merged . در تنظیمات پیش‌فرض فقط 50,000 نمونه به‌صورت streaming به NAAM اضافه می‌شود؛ تکرارهای دقیق با hash حذف می‌شوند. چون ممکن است با منابعی که قبلاً در NAAM آمده‌اند هم‌پوشانی داشته باشد، این dedup فقط تکرارهای دقیقِ موجود در نمونهٔ انتخاب‌شده را می‌گیرد و تضمین حذف همهٔ هم‌پوشانی‌ها نیست. مقدار `merged_rows` را در GitHub می‌توان 0 کرد یا افزایش داد. پیش از انتشار عمومی مدل، منشأ و مجوز منابع تشکیل‌دهنده را هم بررسی کنید.

### SFT اختیاری

`amiralifirouzi/Persian-SFT-102K` شامل نمونه‌های instruction/conversation فارسی است؛ کارت آن در حال حاضر license را `other` نشان می‌دهد، نه یک مجوز استاندارد روشن. بنابراین دریافت و استفاده از آن **به‌صورت پیش‌فرض خاموش است**. فقط پس از بررسی مجوز و شرایط استفاده، گزینهٔ `run_sft` را در GitHub فعال کنید. صفحه: https://huggingface.co/datasets/amiralifirouzi/Persian-SFT-102K

### دادهٔ خصوصی شما

هیچ اسکریپتی در این پروژه پوشهٔ دیتاست خصوصی شما را نمی‌خواند یا آپلود نمی‌کند. دادهٔ خصوصی را به Git، GitHub Actions، Hugging Face یا Artifactها اضافه نکنید. `.gitignore` نیز مسیرهای داده و checkpoint را مستثنا می‌کند.

## ۳) نصب محلی در ویندوز

Python 3.11 پیشنهاد می‌شود. در PowerShell:

```powershell
cd C:\Path\To\Danory_GPT_v7
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

برای آماده‌سازی دادهٔ عمومی (نیازمند اینترنت و فضای دیسک):

```powershell
python scripts\prepare_public_data.py --naab-train 300000 --naab-test 20000 --merged-rows 50000 --train-shards 0 --test-shards 2
```

ساخت توکنایزر 16k:

```powershell
python build_tokenizer.py --input data/public/train.jsonl --out data/tokenizer --vocab-size 16000 --max-chars 200000000
```

بررسی و تست:

```powershell
python scripts\model_info.py
python scripts\preflight.py
python scripts\smoke_test.py
```

آموزش از ابتدا:

```powershell
python train.py --stage pretrain --steps-this-run 100
```

در اجرای محلی، `max_steps` در `configs/train.json` طول کل برنامهٔ زمان‌بندی نرخ یادگیری است. `--steps-this-run` فقط تعداد گام‌های همین اجرا را محدود می‌کند؛ برای ادامه، از checkpoint همان v7 استفاده کنید:

```powershell
python train.py --stage pretrain --resume checkpoints\pretrain_last.pt --steps-this-run 100
```

تولید متن پس از ساخته‌شدن checkpoint:

```powershell
python generate.py --checkpoint checkpoints\pretrain_last.pt --prompt "زبان فارسی یکی از زبان‌های مهم جهان است که" --tokens 160
```

## ۴) راه‌اندازی در GitHub Actions

### ایجاد مخزن

1. در GitHub یک repository جدید بسازید؛ برای استفاده از runner استاندارد رایگان، repository عمومی باشد.
2. ZIP را از حالت فشرده خارج کنید.
3. تمام محتویات پوشهٔ `Danory_GPT_v7` را در ریشهٔ مخزن قرار دهید؛ دقت کنید `.github/workflows/train.yml` هم آپلود شود.
4. در GitHub وارد تب **Actions** شوید و در صورت درخواست، Actions را فعال کنید.
5. در فهرست workflowها، **Danory GPT v7 training** را انتخاب کنید و **Run workflow** را بزنید.

### اجرای اول

- `steps_this_run`: مقدار 1200
- `naab_train_rows`: مقدار 300000
- `naab_test_rows`: مقدار 20000
- `naab_train_shards`: مقدار 0
- `naab_test_shards`: مقدار 2
- `merged_rows`: مقدار 50000
- `resume_run_id`: خالی بماند
- `run_sft`: خاموش بماند تا ابتدا مجوز دیتاست SFT بررسی شود.

اجرای اول، دادهٔ عمومی را stream می‌کند، توکنایزر 16k را می‌سازد، تست مدل را انجام می‌دهد و آموزش را از وزن‌های تصادفی شروع می‌کند. در پایان، Artifact با نام `danory-v7-resume` شامل checkpoint ادامهٔ آموزش و همان tokenizer ایجاد می‌شود. برای محدود ماندن حجم Artifact، فقط checkpoint پیش‌آموزش و tokenizer در آن قرار می‌گیرند؛ retention یک روز است، پس آن را زود دانلود یا برای اجرای بعدی استفاده کنید.

### ادامهٔ آموزش

1. وارد اجرای قبلی در تب Actions شوید و Run ID آن را از صفحهٔ اجرا بردارید.
2. دوباره **Run workflow** را اجرا کنید.
3. `resume_run_id` را با Run ID قبلی پر کنید.
4. مقدار `steps_this_run` را مثلاً 1200 بگذارید.

در این روش، checkpoint و tokenizer دقیقاً از اجرای قبلی بازیابی می‌شوند و دادهٔ عمومیِ آماده‌شده نیز در صورت موجود بودن از Cache Actions بازیابی می‌شود و نرخ یادگیری همان برنامهٔ کلی را ادامه می‌دهد. Artifact این workflow یک روز نگهداری می‌شود تا سهمیهٔ فضای ذخیره‌سازی رایگان با اجرای چندباره پر نشود؛ پس از هر اجرا checkpoint را دانلود کنید یا اجرای بعدی را به‌موقع با Run ID قبلی آغاز کنید.

### افزایش طول کل آموزش

قبل از اولین اجرا، مقدار `max_steps` در `configs/train.json` به‌صورت پیش‌فرض 10000 است؛ در صورت نیاز آن را بیشتر کنید و آن تغییر را commit کنید. سپس هر بار با `steps_this_run=1200` اجرا را ادامه دهید و Run ID اجرای قبل را وارد کنید. **در میانهٔ آموزش، `max_steps` را تغییر ندهید**؛ چون شکل برنامهٔ cosine learning-rate عوض می‌شود. اگر واقعاً می‌خواهید برنامهٔ جدیدی آغاز کنید، از ابتدا آموزش تازه شروع کنید.

### SFT پس از پیش‌آموزش

SFT در این پروژه یک مرحلهٔ جداگانه است و به‌صورت پیش‌فرض در GitHub workflow اجرا نمی‌شود. فقط بعد از بررسی مجوز دیتاست، آن را روی سیستم خود آماده کنید:

```powershell
python scripts\prepare_public_data.py --naab-train 300000 --naab-test 20000 --merged-rows 50000 --train-shards 0 --test-shards 2 --include-sft
python train.py --config configs\sft.json --stage sft --steps-this-run 300
```

در این مرحله وزن‌های checkpoint پیش‌آموزش همین v7 بارگذاری می‌شود و optimizer تازه‌ای برای SFT ساخته می‌شود. این انتقال درون همین v7 است و هیچ ارتباطی با انتقال وزن از v6 یا نسخه‌های قبلی ندارد. برای ادامهٔ SFT در همان برنامهٔ 1000 مرحله‌ای:

```powershell
python train.py --config configs\sft.json --stage sft --resume checkpoints\sft_last.pt --steps-this-run 300
```

SFT dataset license را `other` نشان می‌دهد؛ تا وقتی شرایط آن روشن نشده، از این مرحله صرف‌نظر کنید.

## ۵) محدودیت‌های GitHub رایگان

- runner استاندارد Ubuntu برای repository عمومی طبق مستندات فعلی GitHub دارای 4 هستهٔ CPU، 16GB RAM و 14GB SSD است؛ GPU ندارد.
- هر job در runner میزبانی‌شده حداکثر 6 ساعت اجرا می‌شود.
- پس آموزش به قطعه‌های جدا تقسیم شده و با checkpoint ادامه پیدا می‌کند. هیچ تضمینی نیست که 1200 گام در هر job حتماً در 6 ساعت تمام شود؛ زمان واقعی به runner، طول داده و سرعت شبکه بستگی دارد.
- کل NAAM را نمی‌توان در فضای 14GB ذخیره کرد؛ به همین دلیل streaming و سقف تعداد نمونه استفاده شده است.
- برای مخزن خصوصی، runner استاندارد معمولاً منابع کمتری دارد و مصرف Actions تابع سهمیه/هزینهٔ حساب است.
- Artifact در این پروژه فقط checkpoint پیش‌آموزش و tokenizer را نگه می‌دارد؛ دادهٔ خصوصی کاربر هیچ‌گاه وارد Artifact نمی‌شود. فایل SFT checkpoint به‌صورت خودکار در Artifact قرار نمی‌گیرد؛ SFT را جداگانه روی سیستم خود اجرا کنید و checkpoint آن را در صورت نیاز نگه دارید.

## ۶) فایل‌های مهم

```text
Danory_GPT_v7/
├── .github/workflows/train.yml
├── configs/model.json
├── configs/train.json
├── configs/generation.json
├── danory/model.py
├── danory/data.py
├── danory/trainer.py
├── danory/checkpoint.py
├── scripts/prepare_public_data.py
├── scripts/model_info.py
├── scripts/preflight.py
├── scripts/smoke_test.py
├── build_tokenizer.py
├── train.py
├── generate.py
└── requirements.txt
```

## ۷) نکات علمی و ارزیابی

- کاهش validation loss به‌تنهایی تضمین‌کنندهٔ متن خوب نیست؛ چند prompt ثابت فارسی را در هر checkpoint ارزیابی کنید.
- نمونه‌های NAAM از چند دامنه و سبک آمده‌اند؛ پاک‌سازی پایه و حذف تکرار دقیق در آماده‌سازی انجام می‌شود، اما این کار جایگزین پالایش انسانی یا تشخیص همهٔ محتوای کم‌کیفیت نیست.
- 300k پاراگراف فقط نقطهٔ شروع قابل‌کنترل است. افزایش آن، زمان stream، اندازهٔ فایل، زمان tokenizer و آموزش را افزایش می‌دهد.
- از نظر ظرفیت سخت‌افزاری، 23.45M پارامتر با 1024 context برای runner CPU یک آزمایش قابل‌اجراست، اما برای کیفیت رقابتی به داده و محاسبات بسیار بیشتری نیاز دارد.
