import os
import re
import json
import time
import random
import argparse
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont, features

# -------------------- Tibetan helpers --------------------

def clean_text(text: str) -> str:
    text = re.sub(r'[^\u0F00-\u0FFF་།\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def split_to_syllables(text: str):
    """切成音节列表（以 '་' 或 '།' 作为结束符）"""
    text = clean_text(text)
    syllables = []
    cur = ""
    for c in text:
        cur += c
        if c in ['་', '།']:
            if cur.strip():
                syllables.append(cur.strip())
            cur = ""
    if cur.strip():
        syllables.append(cur.strip())
    return syllables

def list_font_files(font_path: str):
    if os.path.isdir(font_path):
        out = []
        for name in os.listdir(font_path):
            if name.lower().endswith((".ttf", ".otf")):
                out.append(os.path.join(font_path, name))
        return out
    if os.path.isfile(font_path) and font_path.lower().endswith((".ttf", ".otf")):
        return [font_path]
    return []

def list_bg_images(images_dir: str):
    exts = (".png", ".jpg", ".jpeg")
    return [os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(exts)]

# -------------------- Generator core --------------------

class BatchGenerator:
    def __init__(self, args):
        self.images_dir = args.images_dir
        self.texts_dir = args.texts_dir
        self.output_dir = args.output_dir
        self.font_path = args.font_path
        self.num_images = args.num_images
        self.max_workers = args.max_workers
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.allow_cycle = args.allow_cycle

        # 输出
        os.makedirs(self.output_dir, exist_ok=True)
        self.labels_path = os.path.join(self.output_dir, "labels.json")

        # 随机参数范围（你可以按需要改）
        self.font_size_min, self.font_size_max = args.font_size_min, args.font_size_max
        self.line_spacing_min, self.line_spacing_max = args.line_spacing_min, args.line_spacing_max
        self.margin_min, self.margin_max = args.margin_min, args.margin_max
        self.fill_ratio_min, self.fill_ratio_max = args.fill_ratio_min, args.fill_ratio_max

        self.text_color = args.text_color

        # cache
        self.font_cache = {}
        self.font_cache_lock = threading.Lock()
        self.bg_cache = []

        # 全局音节流
        self.all_syllables = []
        self.syllable_ptr = 0
        self.ptr_lock = threading.Lock()

        if self.seed is not None:
            random.seed(self.seed)

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def load_font_cached(self, font_file, size):
        key = (font_file, size)
        with self.font_cache_lock:
            if key not in self.font_cache:
                try:
                    self.font_cache[key] = ImageFont.truetype(font_file, size)
                except Exception:
                    return None
            return self.font_cache[key]

    def preload_backgrounds(self, image_paths, max_cache=120):
        pick = image_paths
        if len(pick) > max_cache:
            pick = random.sample(image_paths, max_cache)

        cached = []
        for p in pick:
            try:
                with Image.open(p) as im:
                    cached.append(im.convert("RGB"))
            except Exception:
                pass
        self.bg_cache = cached
        self.log(f"预加载背景图：{len(cached)}/{len(image_paths)}")

    def load_all_text(self):
        buf = []
        for fn in os.listdir(self.texts_dir):
            if fn.lower().endswith(".txt"):
                p = os.path.join(self.texts_dir, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        buf.append(f.read())
                except Exception:
                    pass
        all_text = "".join(buf)
        self.all_syllables = split_to_syllables(all_text)
        self.log(f"总音节数：{len(self.all_syllables)}")

    # ---------- Capacity: 关键：字大放少，字小放多 ----------

    def _text_width(self, draw, text, font):
        try:
            return int(draw.textlength(text, font=font))
        except Exception:
            bb = draw.textbbox((0, 0), text, font=font)
            return int(bb[2] - bb[0])

    def estimate_capacity(self, image, font, margin, line_spacing):
        """估算在该图+该font+行距下最多能塞多少音节（按宽度装箱+高度行数）"""
        draw = ImageDraw.Draw(image)
        avail_w = max(1, image.width - 2 * margin)
        avail_h = max(1, image.height - 2 * margin)

        line_h = max(1, int(font.size * line_spacing))
        max_lines = max(1, avail_h // line_h)

        # 用代表性音节估算宽度（越大 => 一行能放的越少）
        sample = "བོད་"
        sample_w = max(1, self._text_width(draw, sample, font))
        est_sylls_per_line = max(1, avail_w // sample_w)

        # 这是粗估上限；实际装箱会更准确，但用于确定“段长范围”很够用
        rough_total = max_lines * est_sylls_per_line
        return max_lines, est_sylls_per_line, rough_total

    def take_syllables(self, count):
        """从全局音节流按顺序取 count 个音节：默认不重复；allow_cycle=True 时循环"""
        with self.ptr_lock:
            if not self.all_syllables:
                return ["་"] * count

            start = self.syllable_ptr
            end = start + count

            if end <= len(self.all_syllables):
                seg = self.all_syllables[start:end]
                self.syllable_ptr = end
                return seg

            # 不够了
            if not self.allow_cycle:
                # 只取剩余的（可能不足 count）
                seg = self.all_syllables[start:]
                self.syllable_ptr = len(self.all_syllables)
                return seg

            # 循环取
            seg = []
            idx = start
            for _ in range(count):
                seg.append(self.all_syllables[idx % len(self.all_syllables)])
                idx += 1
            self.syllable_ptr = idx % len(self.all_syllables)
            return seg

    def layout_and_draw(self, image, syllables, font, margin, line_spacing):
        """按宽度装箱排版并绘制；返回 drawn_lines(str list) 和 syllable_boxes"""
        draw = ImageDraw.Draw(image)
        avail_w = max(1, image.width - 2 * margin)
        avail_h = max(1, image.height - 2 * margin)

        line_h = max(1, int(font.size * line_spacing))
        # --- 修改点：不再随机，直接填满物理空间支持的最大行数 ---
        max_lines = max(1, avail_h // line_h)
        use_lines = max_lines

        lines = []
        idx = 0
        # 逐行装箱
        for _ in range(use_lines):
            w = 0
            line_sylls = []
            placed = 0
            while idx < len(syllables):
                syl = syllables[idx]
                syl_w = self._text_width(draw, syl, font)
                if placed == 0 or w + syl_w <= avail_w:
                    line_sylls.append(syl)
                    w += syl_w
                    placed += 1
                    idx += 1
                else:
                    break
            if not line_sylls: break
            lines.append(line_sylls)
            if idx >= len(syllables): break

        # 重新计算整体高度以进行垂直居中（或置顶）
        line_strs = [''.join(line) for line in lines]
        block_h = line_h * len(line_strs)

        # 起始坐标：x居中，y尽量靠顶（留出margin）
        x0 = margin
        y0 = (image.height - block_h) // 2  # 垂直居中，若想彻底填满可改为 y0 = margin

        syllable_boxes = []
        for li, line in enumerate(lines):
            y = y0 + li * line_h
            x = (image.width - self._text_width(draw, line_strs[li], font)) // 2  # 每行水平居中
            for pi, syl in enumerate(line):
                bbox = draw.textbbox((x, y), syl, font=font)
                draw.text((x, y), syl, font=font, fill=self.text_color)

                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(image.width - 1, x2); y2 = min(image.height - 1, y2)

                syllable_boxes.append({
                    "line": li,
                    "pos": pi,
                    "text": syl,
                    "bbox": [x1, y1, x2, y2],
                    "font_size": int(font.size),
                    "margin_used": int(margin),
                    "line_spacing": float(line_spacing),
                })

                x += self._text_width(draw, syl, font)
                if x >= image.width - margin:
                    break

        return line_strs, syllable_boxes

    def generate_one(self, i, font_files):
        """生成单图：随机参数 + 自动段长"""
        if not self.bg_cache:
            return None

        image = random.choice(self.bg_cache).copy()

        # 每张图随机参数
        font_file = random.choice(font_files)
        font_size = random.randint(self.font_size_min, self.font_size_max)
        line_spacing = random.uniform(self.line_spacing_min, self.line_spacing_max)
        margin = random.randint(self.margin_min, self.margin_max)

        font = self.load_font_cached(font_file, font_size)
        if font is None:
            return None

        # 容量估算：字大 => rough_total 小；字小 => rough_total 大
        max_lines, est_spl, rough_total = self.estimate_capacity(image, font, margin, line_spacing)

        # --- 修改点：为了确保填满，取词量增加 20% 的冗余 ---
        target = int(rough_total * 1.2)

        # 从全局音节流取对应长度
        seg = self.take_syllables(target)
        if not seg:
            return None

        # 排版绘制（实际可能用不完 seg，这是正常的；也会造成“字小放更多”的效果）
        drawn_lines, syllable_boxes = self.layout_and_draw(image, seg, font, margin, line_spacing)

        # 输出
        fn = f"{i:06d}.png"
        out_path = os.path.join(self.output_dir, fn)
        image.save(out_path, optimize=False, compress_level=1)

        record = {
            "index": str(i),
            "image": fn,
            "conversations": [
                {"from": "human", "value": "<image>\nOCR"},
                {"from": "gpt", "value": "\n".join(drawn_lines), "value_v1": None}
            ],
            "syllable_boxes": syllable_boxes,
            "meta": {
                "font": os.path.basename(font_file),
                "font_size": font_size,
                "line_spacing": line_spacing,
                "margin": margin,
                "rough_capacity": rough_total,
                "fill_ratio": 1.2,  # 固定为 1.2 以反映 20% 冗余
                "target_syllables": target,
                "used_lines": len(drawn_lines),
            }
        }
        return record

    def run(self):
        self.log(f"libraqm 支持：{features.check('raqm')}")
        font_files = list_font_files(self.font_path)
        if not font_files:
            raise RuntimeError("字体路径无可用 ttf/otf")

        bg_paths = list_bg_images(self.images_dir)
        if not bg_paths:
            raise RuntimeError("背景路径无图片")

        self.load_all_text()
        if not self.all_syllables:
            raise RuntimeError("文本为空或清理后无藏文音节")

        self.preload_backgrounds(bg_paths, max_cache=120)

        # 初始化 labels.json（JSONL）
        with open(self.labels_path, "w", encoding="utf-8") as f:
            pass

        t0 = time.time()
        self.log(f"开始生成：{self.num_images} 张 | workers={self.max_workers} batch={self.batch_size}")

        records_buffer = []
        done = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = []
            for batch_start in range(0, self.num_images, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_images)

                futures.clear()
                for i in range(batch_start, batch_end):
                    futures.append(ex.submit(self.generate_one, i, font_files))

                for fu in as_completed(futures):
                    rec = fu.result()
                    if rec:
                        records_buffer.append(rec)
                    done += 1

                # 批量写 JSONL
                if records_buffer:
                    records_buffer.sort(key=lambda x: int(x["index"]))
                    with open(self.labels_path, "a", encoding="utf-8") as f:
                        for r in records_buffer:
                            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
                    records_buffer.clear()

                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                self.log(f"进度：{done}/{self.num_images} | {speed:.2f} 张/秒")

        elapsed = time.time() - t0
        self.log(f"完成：{self.num_images} 张 | 耗时 {elapsed:.2f}s | 平均 {self.num_images/elapsed:.2f} 张/秒")
        self.log(f"labels: {self.labels_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--texts_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--font_path", required=True)

    ap.add_argument("--num_images", type=int, default=1000)
    ap.add_argument("--max_workers", type=int, default=min(32, (os.cpu_count() or 1) * 4))
    ap.add_argument("--batch_size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow_cycle", action="store_true", help="文本不够时循环使用")

    # 随机范围
    ap.add_argument("--font_size_min", type=int, default=20)
    ap.add_argument("--font_size_max", type=int, default=42)
    ap.add_argument("--line_spacing_min", type=float, default=1.10)
    ap.add_argument("--line_spacing_max", type=float, default=1.55)
    ap.add_argument("--margin_min", type=int, default=20)
    ap.add_argument("--margin_max", type=int, default=60)
    ap.add_argument("--fill_ratio_min", type=float, default=0.65)
    ap.add_argument("--fill_ratio_max", type=float, default=1.00)

    ap.add_argument("--text_color", default="#2C2C2C")
    return ap.parse_args()

if __name__ == "__main__":
    # 如果没有提供命令行参数，使用默认参数运行
    if len(sys.argv) == 1:
        # 直接设置默认参数进行运行
        class Args:
            images_dir = r"D:\code\自动化合成数据\1合成图片\jinwen"
            texts_dir = r"D:\code\自动化合成数据\1合成图片\txt"
            output_dir = r"D:\code\古籍数据合成代码\text1"
            font_path = r"D:\code\自动化合成数据\1合成图片\fonts2"
            num_images = 10           # 生成的图片总数
            max_workers = min(32, (os.cpu_count() or 1) * 4)  # 最大工作线程数（不超过32）
            batch_size = 500                   # 批处理大小，每次处理的图片数量
            seed = None                        # 随机种子（None表示每次运行使用不同的随机种子）
            allow_cycle = True                 # 是否允许循环使用文本（当文本不够时）
            font_size_min = 30                 # 字体大小最小值（像素）
            font_size_max = 80                 # 字体大小最大值（像素）
            line_spacing_min = 1.10            # 行间距最小倍数（相对于字体大小）
            line_spacing_max = 1.55            # 行间距最大倍数（相对于字体大小）
            margin_min = 20                    # 边距最小值（像素）
            margin_max = 50                    # 边距最大值（像素），调小以便文字贴近边框
            fill_ratio_min = 1.00              # 文本填充比例最小值（0.0-1.0），锁定为 1.0
            fill_ratio_max = 1.00              # 文本填充比例最大值（0.0-1.0），锁定为 1.0
            text_color = "#2C2C2C"             # 文字颜色（深灰色）

        args = Args()
    else:
        args = parse_args()

    # 先提取input数据
    # def extract_inputs_from_jsonl():
    #     jsonl_path = r"D:\code\自动化合成数据\合成图片1\txt1\Tibetan monolingual data.jsonl"
    #     output_txt_path = r"D:\code\自动化合成数据\合成图片1\txt1\extracted_inputs.txt"
    #
    #     inputs = []
    #     print(f"开始处理JSONL文件: {jsonl_path}")
    #
    #     with open(jsonl_path, 'r', encoding='utf-8') as f:
    #         for line_num, line in enumerate(f, 1):
    #             line = line.strip()
    #             if not line:
    #                 continue
    #
    #             # 使用正则表达式提取input字段的值
    #             match = re.search(r'"input"\s*:\s*"([^"]*)"', line)
    #             if match:
    #                 input_text = match.group(1)
    #                 inputs.append(input_text)
    #
    #             # 每处理10000行显示一次进度
    #             if line_num % 10000 == 0:
    #                 print(f"已处理 {line_num} 行，提取了 {len(inputs)} 个input")
    #
    #     print(f"总共提取了 {len(inputs)} 个input")
    #
    #     # 保存到txt文件
    #     print(f"保存到文件: {output_txt_path}")
    #     with open(output_txt_path, 'w', encoding='utf-8') as f:
    #         for input_text in inputs:
    #             f.write(input_text + '\n')
    #
    #     print("Input提取完成！")
    #     return inputs

    # 运行input提取
    #inputs = extract_inputs_from_jsonl()

    # 然后运行图片生成
    gen = BatchGenerator(args)
    gen.run()
