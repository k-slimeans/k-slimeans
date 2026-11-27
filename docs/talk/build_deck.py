#!/usr/bin/env python3
"""Build the talk deck (.pptx) from docs/talk/deck.yaml.

    build_deck.py --content docs/talk/deck.yaml --out deck.pptx
                  [--template TEMPLATE.pptx|.potx] [--only 1.4,3.6]
    build_deck.py --template TEMPLATE.pptx --print-layouts

Without --template the deck is drawn on python-pptx's default template with a
Missouri S&T look (fallback mode). With --template the slides are created from
the template's own layouts (template mode): titles go into the layout's title
placeholder and every other element is laid out inside the area under it.

Needs python-pptx, PyYAML and lxml (Pillow, which python-pptx pulls in, is used
to measure text when the fonts are installed).
"""

import argparse
import copy
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from collections import namedtuple
from decimal import ROUND_HALF_UP, Decimal

import yaml
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

try:
    from PIL import ImageFont
except ImportError:  # measuring then falls back to a flat average glyph width
    ImageFont = None

# --------------------------------------------------------------------------- palette

FOREST, MINER, LIMA, KIWI = "154734", "007A33", "72BF44", "BFD730"
ORIENT, LAGOON, SHUTTLE, PEBBLE = "005F83", "00858A", "63666A", "D6D1CA"
TANGO, RUBY, TINT, TANGO_TINT = "E77724", "AD2327", "EEF4EF", "FBE3D0"
INK, WHITE = "1B1B1B", "FFFFFF"
AMETHYST = "403A60"
PALETTE = {
    "forest": FOREST, "amethyst": AMETHYST, "miner": MINER, "lima": LIMA, "kiwi": KIWI, "orient": ORIENT, "lagoon": LAGOON,
    "shuttle": SHUTTLE, "pebble": PEBBLE, "tango": TANGO, "ruby": RUBY,
}

Box = namedtuple("Box", "x y w h")
Box.r = property(lambda b: b.x + b.w)
Box.b = property(lambda b: b.y + b.h)

SECTION_NAMES = {
    "1": "1 Question & setup (Cooper)",
    "2": "2 Related work (Samiksha, Nabin)",
    "3": "3 What a replacement does (Cooper)",
    "4": "4 Summary quality (Samiksha)",
    "5": "5 Case studies (Samiksha)",
    "6": "6 Model size (Nabin)",
    "7": "7 Cost (Nabin)",
    "8": "8 Answer (Cooper)",
    "backup": "Backup",
}

MINUS = "−"
LINE = 1.22  # line height as a multiple of the font size (Calibri)


def I(v):
    """Inches to EMU."""
    return Emu(int(round(v * 914400)))


def rgb(hexstr):
    return RGBColor.from_string(hexstr)


# --------------------------------------------------------------------------- text measuring

_FONT_FILES = {
    "body": [
        ("/mnt/c/Windows/Fonts/calibri.ttf", "/mnt/c/Windows/Fonts/calibrib.ttf",
         "/mnt/c/Windows/Fonts/calibrii.ttf", "/mnt/c/Windows/Fonts/calibriz.ttf"),
        ("/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf", "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
         "/usr/share/fonts/truetype/crosextra/Carlito-Italic.ttf", "/usr/share/fonts/truetype/crosextra/Carlito-BoldItalic.ttf"),
    ],
    "head": [
        ("/mnt/c/Windows/Fonts/framd.ttf", "/mnt/c/Windows/Fonts/framd.ttf",
         "/mnt/c/Windows/Fonts/framdit.ttf", "/mnt/c/Windows/Fonts/framdit.ttf"),
    ],
    "eq": [
        ("/mnt/c/Windows/Fonts/cambria.ttc", "/mnt/c/Windows/Fonts/cambriab.ttf",
         "/mnt/c/Windows/Fonts/cambriai.ttf", "/mnt/c/Windows/Fonts/cambriaz.ttf"),
        ("/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf", "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf",
         "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf", "/usr/share/fonts/truetype/crosextra/Caladea-BoldItalic.ttf"),
    ],
}
_font_cache = {}
_width_cache = {}
REF_PX = 100


def _load_font(key, bold, italic):
    ck = (key, bold, italic)
    if ck in _font_cache:
        return _font_cache[ck]
    font = None
    if ImageFont is not None:
        idx = (1 if bold else 0) + (2 if italic else 0)
        order = [idx, 1 if bold else 0, 0]
        for family in _FONT_FILES[key] + _FONT_FILES["body"]:
            for i in order:
                path = family[i]
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, REF_PX, index=0)
                    except Exception:
                        font = None
                    if font:
                        break
            if font:
                break
    _font_cache[ck] = font
    return font


def char_em(ch, key, bold, italic):
    """Advance width of one character in em."""
    ck = (ch, key, bold, italic)
    v = _width_cache.get(ck)
    if v is None:
        font = _load_font(key, bold, italic)
        v = font.getlength(ch) / REF_PX if font else 0.5
        _width_cache[ck] = v
    return v


# --------------------------------------------------------------------------- paragraph model

_TOK = re.compile(r"(\*\*.+?\*\*|_\{.+?\}|\^\{.+?\})")
_UNIT_GAP = re.compile(r"(?<=\d) (?=(?:s|ms|GB|df|min)\b)")  # keep a number and its unit on one line
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": MINUS}


def parse(text, eq=False):
    """Split markup into (text, style) segments: **bold**, and in equations _{sub} and ^{sup}."""
    segs = []
    text = _UNIT_GAP.sub("\u00a0", text)
    for part in _TOK.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            segs.append((part[2:-2], {"bold": True}))
        elif eq and part.startswith("_{") and part.endswith("}"):
            segs.append((part[2:-1], {"base": "sub"}))
        elif eq and part.startswith("^{") and part.endswith("}"):
            segs.append((part[2:-1], {"base": "sup"}))
        else:
            segs.append((part, {}))
    return segs


def stat_text(s):
    """Rewrite unicode superscript runs (10⁻¹⁶) as ^{...} markup so any font can show them."""
    return re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^{" + "".join(_SUP[ch] for ch in m.group(0)) + "}", s)


class P:
    """One paragraph: text with markup plus its look."""

    def __init__(self, text, size=16, bold=False, italic=False, color=INK, font="body", align="l", bullet=False,
                 after=0, before=0, eq=False, spc=None, line=None):
        self.text, self.size, self.bold, self.italic, self.color = text, size, bold, italic, color
        self.font, self.align, self.bullet, self.after, self.before = font, align, bullet, after, before
        self.eq, self.spc, self.line = eq, spc, line


BULLET_INDENT = 0.28


def para_lines(p, width):
    """Greedy word wrap; returns the number of lines the paragraph takes in `width` inches."""
    if p.bullet:
        width -= BULLET_INDENT
    chars = []
    for text, st in parse(p.text, p.eq):
        scale = 0.65 if st.get("base") else 1.0
        bold = p.bold or st.get("bold", False)
        for ch in text:
            chars.append((ch, char_em(ch, p.font, bold, p.italic) * p.size * scale / 72.0 * 1.02, ch == " "))
    lines, cur, word = 1, 0.0, 0.0
    pending_space = 0.0

    def flush():
        nonlocal lines, cur, word, pending_space
        if word == 0.0:
            return
        if cur > 0 and cur + pending_space + word > width:
            lines += 1
            cur = 0.0
            pending_space = 0.0
        if word > width:
            lines += int(word // width)
            word = word % width
        cur += pending_space + word
        word, pending_space = 0.0, 0.0

    for ch, w, is_space in chars:
        if is_space:
            flush()
            pending_space += w
        else:
            word += w
    flush()
    return lines


def longest_word(p):
    """Width in inches of the longest unbreakable word in the paragraph."""
    best = 0.0
    for word in re.split(r"\s+", "".join(t for t, _ in parse(p.text, p.eq))):
        best = max(best, sum(char_em(ch, p.font, p.bold, p.italic) * p.size / 72.0 * 1.02 for ch in word))
    return best


def para_height(p, width):
    step = (p.line if p.line else p.size * LINE) / 72.0
    return para_lines(p, width) * step + (p.before + p.after) / 72.0


def text_height(paras, width):
    return sum(para_height(p, width) for p in paras)


def max_line_width(p):
    return sum(char_em(ch, p.font, p.bold, p.italic) * p.size / 72.0 * 1.02 for ch in "".join(t for t, _ in parse(p.text, p.eq)))


# --------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, mode, content, prs):
        self.mode = mode
        self.content = content
        self.prs = prs
        self.warnings = []
        self.W = prs.slide_width / 914400.0
        self.H = prs.slide_height / 914400.0
        if mode == "template":
            self.face = {"body": "+mn-lt", "head": "+mj-lt", "eq": "Cambria Math"}
        else:
            self.face = {"body": "Calibri", "head": "Franklin Gothic Medium", "eq": "Cambria Math"}
        self.colors = {k: PALETTE[v] for k, v in (content.get("series_colors") or {}).items()}
        self.layouts = {}
        self.slide_id = ""

    def warn(self, msg):
        self.warnings.append(f"slide {self.slide_id}: {msg}")


# --------------------------------------------------------------------------- low-level drawing

_ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
_ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


def write_para(c, para, p):
    para.alignment = _ALIGN[p.align]
    if p.line:
        para.line_spacing = Pt(p.line)
    para.space_before = Pt(p.before)
    para.space_after = Pt(p.after)
    for text, st in parse(p.text, p.eq):
        r = para.add_run()
        r.text = text
        f = r.font
        f.size = Pt(p.size)
        f.bold = bool(p.bold or st.get("bold"))
        f.italic = bool(p.italic)
        f.name = c.face[p.font]
        if p.color:
            f.color.rgb = rgb(p.color)
        rpr = r._r.get_or_add_rPr()
        if st.get("base") == "sub":
            rpr.set("baseline", "-25000")
        elif st.get("base") == "sup":
            rpr.set("baseline", "30000")
        if p.spc:
            rpr.set("spc", str(p.spc))
    if p.bullet:
        ppr = para._p.get_or_add_pPr()
        ppr.set("marL", str(int(BULLET_INDENT * 914400)))
        ppr.set("indent", str(-int(BULLET_INDENT * 914400)))
        clr = etree.SubElement(ppr, qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", MINER)
        etree.SubElement(ppr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(ppr, qn("a:buChar")).set("char", "•")


def fill_tf(c, tf, paras, anchor="t", ins=(0.0, 0.0, 0.0, 0.0)):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (I(v) for v in ins)
    tf.vertical_anchor = _ANCHOR[anchor]
    for i, p in enumerate(paras):
        write_para(c, tf.paragraphs[0] if i == 0 else tf.add_paragraph(), p)


def drop_style(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def add_text(c, sl, x, y, w, h, paras, name, anchor="t", ins=(0.0, 0.0, 0.0, 0.0)):
    tb = sl.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tb.name = name
    fill_tf(c, tb.text_frame, paras, anchor, ins)
    return tb


def add_box(sl, kind, x, y, w, h, name, fill=None, line=None, line_w=1.0, dash=None, radius=None):
    shp = sl.shapes.add_shape(kind, I(x), I(y), I(w), I(h))
    shp.name = name
    drop_style(shp)
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(line_w)
        if dash:
            shp.line.dash_style = dash
    else:
        shp.line.fill.background()
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = min(0.5, radius / max(0.01, min(w, h)))
    return shp


CARD_INS = (0.2, 0.12, 0.2, 0.12)


def add_card(c, sl, x, y, w, h, paras, name, fill=TINT, anchor="t", ins=CARD_INS, line=None, dash=None, radius=0.12):
    shp = add_box(sl, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, name, fill=fill, line=line, line_w=1.25, dash=dash, radius=radius)
    fill_tf(c, shp.text_frame, paras, anchor, ins)
    return shp


def card_inner_w(w, ins=CARD_INS):
    return w - ins[0] - ins[2]


def card_need(paras, w, ins=CARD_INS):
    return text_height(paras, card_inner_w(w, ins)) + ins[1] + ins[3] + 0.04


def add_line(sl, x1, y1, x2, y2, color, width_pt, name, dash=None, arrow=False):
    ln = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    ln.name = name
    drop_style(ln)
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(width_pt)
    if dash:
        ln.line.dash_style = dash
    if arrow:
        etree.SubElement(ln.line._get_or_add_ln(), qn("a:tailEnd")).attrib.update({"type": "triangle", "w": "med", "len": "med"})
    return ln


def fit(make, width, height, sizes, c=None, what=""):
    """Largest size in `sizes` (descending) whose paragraphs fit width x height; (paras, size, fits)."""
    for s in sizes:
        paras = make(s)
        if text_height(paras, width) <= height:
            return paras, s, True
    if c is not None:
        c.warn(f"text does not fit at the minimum size ({what})")
    return make(sizes[-1]), sizes[-1], False


def series_color(c, label):
    return c.colors.get(label, FOREST)


# --------------------------------------------------------------------------- slide frame

def kicker_text(s):
    if s["part"] == "backup":
        k = "BACKUP"
    else:
        k = f"PART {s['part']} · {str(s['speaker']).upper()}"
    if s["type"] == "placeholder":
        k += " · DRAFT"
    return k


def clear_slides(prs):
    lst = prs.slides._sldIdLst
    for sld in list(lst):
        prs.part.drop_rel(sld.rId)
        lst.remove(sld)
    pres = prs.part._element
    for ext in pres.findall(".//" + qn("p:ext")):
        if ext.get("uri") == "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}":
            parent = ext.getparent()
            parent.remove(ext)
            if len(parent) == 0:
                parent.getparent().remove(parent)


def ph_type(ph):
    try:
        return ph.placeholder_format.type
    except Exception:
        return None


TITLE_TYPES = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
SUB_TYPES = (PP_PLACEHOLDER.SUBTITLE, PP_PLACEHOLDER.BODY)


def layout_has(layout, types):
    return any(ph_type(ph) in types for ph in layout.placeholders)


def pick_layouts(prs):
    layouts = [lay for m in prs.slide_masters for lay in m.slide_layouts]
    by_name = {}
    for lay in layouts:
        by_name.setdefault(lay.name.strip().lower(), lay)
    title = by_name.get("title slide") or next(
        (lay for lay in layouts if layout_has(lay, TITLE_TYPES) and layout_has(lay, (PP_PLACEHOLDER.SUBTITLE,))), None)
    content = (by_name.get("title only") or by_name.get("title and content")
               or next((lay for lay in layouts if layout_has(lay, TITLE_TYPES)), None))
    closing = by_name.get("section header") or title
    if title is None:
        title = content
    if content is None or title is None:
        sys.exit("error: the template has no layout with a title placeholder")
    return {"title": title, "content": content, "closing": closing or content}


def print_layouts(prs):
    print(f"slide size: {prs.slide_width / 914400:.3f} x {prs.slide_height / 914400:.3f} in")
    for mi, m in enumerate(prs.slide_masters):
        print(f"master {mi}")
        for lay in m.slide_layouts:
            print(f'  layout "{lay.name}"')
            for ph in lay.placeholders:
                t = ph_type(ph)
                pos = tuple(round((v or 0) / 914400.0, 2) for v in (ph.left, ph.top, ph.width, ph.height))
                print(f"    idx={ph.placeholder_format.idx:<3} type={str(t).split('.')[-1].split(' ')[0]:<14} "
                      f'name="{ph.name}" x,y,w,h={pos}')


def set_title_text(ph, text):
    tf = ph.text_frame
    p = tf.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.add_run().text = text
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def title_size_pt(slide):
    """Font size the title placeholder inherits from its layout or master (32 pt if none is found)."""
    layout = slide.slide_layout
    for root in (layout._element, layout.slide_master._element):
        sz = root.xpath('.//p:sp[p:nvSpPr/p:nvPr/p:ph[@type="title" or @type="ctrTitle"]]//a:lstStyle/a:lvl1pPr/a:defRPr/@sz')
        if sz:
            return int(sz[0]) / 100.0
    sz = layout.slide_master._element.xpath(".//p:txStyles/p:titleStyle/a:lvl1pPr/a:defRPr/@sz")
    return int(sz[0]) / 100.0 if sz else 32.0


def fit_title(c, slide, tph, text):
    """Fill the title placeholder and, if the text would overflow it, store a shrink factor (normAutofit fontScale)."""
    set_title_text(tph, text)
    size = title_size_pt(slide)
    w, h = tph.width / 914400.0 - 0.2, tph.height / 914400.0 - 0.1
    scale = 1.0
    for pct in range(100, 49, -5):
        if text_height([P(text, size=size * pct / 100.0, font="head")], w) <= h:
            scale = pct / 100.0
            break
    else:
        scale = 0.5
    if scale < 1.0:
        tph.text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        tph.text_frame._txBody.bodyPr.find(qn("a:normAutofit")).set("fontScale", str(int(scale * 100000)))
        if scale < 0.7:
            c.warn(f"title shrinks to {int(scale * 100)}% to fit the template's title placeholder")


def remove_empty_placeholders(slide, keep):
    for ph in list(slide.placeholders):
        if ph in keep or ph.has_text_frame and ph.text_frame.text.strip():
            continue
        ph._element.getparent().remove(ph._element)


def new_slide(c, kind):
    """Create a slide; returns (slide, title_placeholder_or_None)."""
    if c.mode == "fallback":
        return c.prs.slides.add_slide(c.prs.slide_layouts[6]), None
    layout = c.layouts[kind]
    sl = c.prs.slides.add_slide(layout)
    return sl, sl.shapes.title


def source_line(c, sl, s, box_left, box_w):
    src = s.get("source")
    if not src:
        return
    if c.mode == "fallback":
        y, h, w = c.H - 0.6, 0.45, box_w - 3.2
    else:
        y, h, w = c.H - 0.72, 0.42, box_w
    add_text(c, sl, box_left, y, w, h, [P(src, size=10, color=SHUTTLE)], "source line", anchor="t" if c.mode == "template" else "b")


def content_frame(c, s, n):
    """Create a content slide with kicker, title, source line and footer; returns (slide, content Box)."""
    sl, tph = new_slide(c, "content")
    if c.mode == "fallback":
        left, width = 0.5, c.W - 1.0
        add_text(c, sl, left, 0.38, 8.0, 0.28, [P(kicker_text(s), size=12, color=MINER, bold=True, spc=150)], "kicker")
        title = s["title"]
        sizes = [28, 27, 26, 25, 24]
        paras, _, _ = fit(lambda z: [P(title, size=z, color=FOREST, font="head")], width, 1.0, sizes, c, "title")
        add_text(c, sl, left, 0.68, width, 1.02, paras, "title")
        box = Box(left, 1.82, width, c.H - 0.75 - 1.82)
        source_line(c, sl, s, left, width)
        add_text(c, sl, c.W - 0.5 - 3.0, c.H - 0.6, 3.0, 0.45,
                 [P(f"{c.content['meta']['footer']}  ·  {n}", size=10, color=SHUTTLE, align="r")], "footer", anchor="b")
        return sl, box
    # template mode
    if tph is None:
        sys.exit("error: content layout has no title placeholder")
    fit_title(c, sl, tph, s["title"])
    tl, tt, tw, th = (v / 914400.0 for v in (tph.left, tph.top, tph.width, tph.height))
    add_text(c, sl, tl, max(0.1, tt - 0.3), min(tw, 8.0), 0.28,
             [P(kicker_text(s), size=12, color=MINER, bold=True, spc=150)], "kicker")
    top = tt + th + 0.15
    box = Box(tl, top, tw, max(1.5, c.H - 0.75 - top))
    remove_empty_placeholders(sl, [tph])
    source_line(c, sl, s, tl, tw)
    return sl, box


def write_notes(sl, s):
    parts = [str(s.get("notes", "")).strip()]
    src = s.get("source") or ("; ".join(s["sources"]) if s.get("sources") else None)
    tail = []
    if src:
        tail.append("Source: " + src)
    tail.append(f"Time: {s['minutes']:g} min")
    sl.notes_slide.notes_text_frame.text = "\n\n".join([parts[0], "\n".join(tail)])


# --------------------------------------------------------------------------- side cards

SIDE_STEPS = {
    "stat": [(32, 16), (30, 15), (28, 14), (26, 14)],
    "quote": [(17, 14), (16, 14), (15, 13), (14, 12)],
    "head": [(17, 16), (16, 15), (15, 14), (14, 14)],
}


def side_kind(item):
    return "stat" if "stat" in item else "quote" if "quote" in item else "head"


def side_paras(item, step):
    kind = side_kind(item)
    a, b = SIDE_STEPS[kind][min(step, len(SIDE_STEPS[kind]) - 1)]
    if kind == "stat":
        return [P(stat_text(item["stat"]), size=a, font="head", color=FOREST, eq=True, after=3),
                P(item["text"], size=b)]
    if kind == "quote":
        return [P(item["quote"], size=a, italic=True, color=FOREST, after=5), P(item["text"], size=b, color=SHUTTLE)]
    return [P(item["head"], size=a, bold=True, color=FOREST, after=3), P(item["text"], size=b)]


def side_fit_step(items, w, h, gap):
    """First step at which the cards stacked in a column of width w fit in height h (None if none)."""
    for step in range(4):
        total = sum(card_need(side_paras(it, step), w) for it in items) + gap * (len(items) - 1)
        if total <= h:
            return step
    return None


def side_column(c, sl, x, y, w, h, items, gap=0.18):
    step = side_fit_step(items, w, h, gap)
    if step is None:
        step = 3
        c.warn("side cards do not fit at the minimum size")
    needs = [card_need(side_paras(it, step), w) for it in items]
    extra = max(0.0, (h - sum(needs) - gap * (len(items) - 1)) / len(items))
    cy = y
    for i, it in enumerate(items):
        ch = needs[i] + extra
        add_card(c, sl, x, cy, w, ch, side_paras(it, step), f"side card {i + 1}", anchor="m")
        cy += ch + gap


def side_row(c, sl, x, y, w, items, step, gap=0.25):
    """Cards side by side at a given size step; returns the row height (drawn only if sl is not None)."""
    n = len(items)
    cw = (w - gap * (n - 1)) / n
    h = max(card_need(side_paras(it, step), cw) for it in items)
    if sl is not None:
        for i, it in enumerate(items):
            add_card(c, sl, x + i * (cw + gap), y, cw, h, side_paras(it, step), f"side card {i + 1}", anchor="m")
    return h


# --------------------------------------------------------------------------- slide types: title / closing

def build_title(c, s, n):
    sl, tph = new_slide(c, "title")
    ext = []
    if c.mode == "fallback":
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = rgb(FOREST)
        left, w = 0.9, c.W - 1.8
        paras, _, _ = fit(lambda z: [P(s["title"], size=z, color=WHITE, font="head")], w, 1.7, [44, 42, 40], c, "title")
        add_text(c, sl, left, 1.0, w, 1.75, paras, "title", anchor="b")
        sub, _, _ = fit(lambda z: [P(s["subtitle"], size=z, color=KIWI)], w, 0.9, [26, 24, 22, 20], c, "subtitle")
        add_text(c, sl, left, 2.95, w, 0.9, sub, "subtitle")
        y0, col_main, col_quote = 4.45, WHITE, PEBBLE
    else:
        if tph is None:
            sys.exit("error: title layout has no title placeholder")
        fit_title(c, sl, tph, s["title"])
        sub_ph = next((ph for ph in sl.placeholders if ph_type(ph) in SUB_TYPES and ph is not tph), None)
        keep = [tph]
        if sub_ph is not None:
            set_title_text(sub_ph, s["subtitle"])
            keep.append(sub_ph)
            sl_, st_, sw_, sh_ = sub_ph.left, sub_ph.top, sub_ph.width, sub_ph.height
            sub_ph.left, sub_ph.top, sub_ph.width = sl_, st_, sw_
            sub_ph.height = min(sh_, I(1.3))  # the subtitle is short; free the space below it
            bottom, left, w = (sub_ph.top + sub_ph.height) / 914400.0, sub_ph.left / 914400.0, sub_ph.width / 914400.0
        else:
            bottom, left, w = (tph.top + tph.height) / 914400.0, tph.left / 914400.0, tph.width / 914400.0
            add_text(c, sl, left, bottom + 0.1, w, 0.6, [P(s["subtitle"], size=22, font="body", color=None)], "subtitle")
            bottom += 0.7
        remove_empty_placeholders(sl, keep)
        y0, col_main, col_quote = bottom + 0.1, None, None
    avail = (c.H - 0.75 - y0) if c.mode == "template" else 2.7
    if c.mode == "template" and avail < 1.0:
        y0, avail = max(0.2, c.H - 0.75 - 1.0), 1.0
    gap_after = 14 if c.mode == "fallback" else 6
    paras_q = lambda z: [P(s["byline"], size=z, color=col_main, after=2), P(s["occasion"], size=z, color=col_main, after=gap_after),
                         P(s["quote"], size=max(14, z - 2), italic=True, color=col_quote)]
    paras_all, _, _ = fit(paras_q, w, avail, [18, 17, 16, 15, 14], c, "title slide text")
    add_text(c, sl, left, y0, w, avail, paras_all, "byline, occasion and quote")
    return sl


def build_closing(c, s, n):
    sl, tph = new_slide(c, "closing")
    if c.mode == "fallback":
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = rgb(FOREST)
        left, w = 0.9, c.W - 1.8
        add_text(c, sl, left, 2.1, w, 1.4, [P(s["title"], size=54, color=WHITE, font="head")], "title", anchor="b")
        sub, _, _ = fit(lambda z: [P(s["subtitle"], size=z, color=KIWI)], w, 0.5, [24, 22, 20, 18], c, "subtitle")
        add_text(c, sl, left, 3.75, w, 1.2, sub, "subtitle")
    else:
        fit_title(c, sl, tph, s["title"])
        sub_ph = next((ph for ph in sl.placeholders if ph_type(ph) in SUB_TYPES and ph is not tph), None)
        keep = [tph]
        if sub_ph is not None:
            set_title_text(sub_ph, s["subtitle"])
            keep.append(sub_ph)
        else:
            tl, tt, tw, th = (v / 914400.0 for v in (tph.left, tph.top, tph.width, tph.height))
            add_text(c, sl, tl, tt + th + 0.1, tw, 0.9, [P(s["subtitle"], size=20, color=None)], "subtitle")
        remove_empty_placeholders(sl, keep)
    return sl


# --------------------------------------------------------------------------- slide types: text

def bullet_paras(items, size, after=None):
    after = size * 0.5 if after is None else after
    return [P(t, size=size, bullet=True, after=after) for t in items]


def build_two_column(c, sl, s, box):
    gap = 0.3
    cw = (box.w - gap) / 2
    cols = [s["left"], s["right"]]
    ins = (0.28, 0.22, 0.28, 0.22)

    def make(z, col):
        return [P(col["heading"], size=z + 3, font="head", color=FOREST, after=10)] + bullet_paras(col["items"], z)

    chosen = None
    for z in (20, 19, 18, 17, 16, 15, 14):
        needs = [card_need(make(z, col), cw, ins) for col in cols]
        if max(needs) <= box.h:
            chosen = z
            break
    if chosen is None:
        chosen = 14
        c.warn("two-column text does not fit at 14 pt")
    h = min(box.h, max(card_need(make(chosen, col), cw, ins) for col in cols) + 0.3)
    h = max(h, 0.55 * box.h)
    for i, col in enumerate(cols):
        add_card(c, sl, box.x + i * (cw + gap), box.y, cw, h, make(chosen, col), f"card {col['heading']}", ins=ins)


def build_answer(c, sl, s, box):
    gap = 0.3
    cw = (box.w - gap) / 2
    cols = [s["left"], s["right"]]
    ins = (0.28, 0.2, 0.28, 0.2)
    limits_h = lambda z: text_height([P(s["limits"], size=z, color=SHUTTLE)], box.w) + 0.05

    def make(z, col):
        return [P(col["heading"], size=z + 3, font="head", color=FOREST, after=8)] + bullet_paras(col["items"], z, after=7)

    chosen, lz = None, 13
    for z in (18, 17, 16, 15, 14):
        for lz in (14, 13, 12):
            avail = box.h - limits_h(lz) - 0.2
            if max(card_need(make(z, col), cw, ins) for col in cols) <= avail:
                chosen = z
                break
        if chosen:
            break
    if chosen is None:
        chosen, lz = 14, 12
        c.warn("answer text does not fit at 14 pt")
    lh = limits_h(lz)
    h = box.h - lh - 0.2
    for i, col in enumerate(cols):
        add_card(c, sl, box.x + i * (cw + gap), box.y, cw, h, make(chosen, col), f"card {col['heading']}", ins=ins)
    add_text(c, sl, box.x, box.y + h + 0.15, box.w, lh, [P(s["limits"], size=lz, color=SHUTTLE)], "limits")


def build_bullets(c, sl, s, box):
    ins = (0.35, 0.28, 0.35, 0.28)
    paras, z, _ = fit(lambda z: bullet_paras(s["items"], z, after=z * 0.7), card_inner_w(box.w, ins), box.h - 0.6,
                      [24, 22, 20, 19, 18, 17, 16, 15, 14], c, "bullets")
    h = min(box.h, max(2.0, card_need(paras, box.w, ins) + 0.2))
    add_card(c, sl, box.x, box.y, box.w, h, paras, "bullet card", ins=ins)


def build_placeholder(c, sl, s, box):
    h = min(box.h, 4.2)
    sources = "Sources: " + ", ".join(s.get("sources", []))
    ins = (0.6, 0.3, 0.6, 0.3)
    paras = [P(f"Draft slide for {s['speaker']}", size=24, bold=True, color=SHUTTLE, after=12),
             P(s["todo"], size=20, color=INK, after=14),
             P(sources, size=14, color=SHUTTLE)]
    add_card(c, sl, box.x, box.y, box.w, h, paras, "draft placeholder", fill=None, anchor="m", ins=ins,
             line=PEBBLE, dash=MSO_LINE.DASH, radius=0.15)


# --------------------------------------------------------------------------- slide type: flow

def build_flow(c, sl, s, box):
    steps = s["steps"]
    n = len(steps)
    sc = box.w / 12.33
    g, pad = 0.42 * sc, 0.12 * sc
    sw = (box.w - (n - 1) * g) / n
    loop = [i for i, t in enumerate(steps) if t.get("loop")]
    levels = [((18, 15), (17, 16), 16), ((17, 14), (16, 15), 15), ((16, 14), (15, 14), 14)]
    ins = (0.1, 0.1, 0.1, 0.1)
    plan = None
    for (lz, dz), (vn, vd), cz in levels:
        step_paras = [[P(t["label"], size=lz, font="head", color=WHITE if t.get("accent") else FOREST, align="c", after=3),
                       P(t["detail"], size=dz, color=WHITE if t.get("accent") else INK, align="c")] for t in steps]
        if any(longest_word(q) > sw - ins[0] - ins[2] for p in step_paras for q in p) and (lz, dz) != levels[-1][0]:
            continue  # a word would be split across lines at this size
        sh = max(card_need(p, sw, ins) for p in step_paras)
        var_paras = [[P(v["name"], size=vn, bold=True, color=FOREST, after=2), P(v["detail"], size=vd)] for v in s["variants"]]
        vw = (box.w - 0.3) / 2
        vh = max(card_need(p, vw) for p in var_paras)
        call = [P(s["callout"], size=cz + 2, bold=True, color=FOREST)]
        chh = card_need(call, box.w, (0.3, 0.12, 0.3, 0.12))
        label_h, ret_h = 0.38, 0.38
        gh = label_h + sh + ret_h + 0.06
        total = gh + vh + chh + 0.5
        plan = (step_paras, sh, var_paras, vw, vh, call, chh, label_h, ret_h, gh)
        if total <= box.h:
            break
    else:
        c.warn("flow slide does not fit at the minimum sizes")
    step_paras, sh, var_paras, vw, vh, call, chh, label_h, ret_h, gh = plan
    total = gh + vh + chh
    gap = min(0.4, max(0.2, (box.h - total) / 2))
    y = box.y
    # loop group
    sy = y + label_h
    if loop:
        gx0 = box.x + loop[0] * (sw + g) - pad
        gx1 = box.x + loop[-1] * (sw + g) + sw + pad
        add_box(sl, MSO_SHAPE.ROUNDED_RECTANGLE, gx0, y, gx1 - gx0, gh, "loop group", fill=TINT, line=PEBBLE, line_w=1.0, radius=0.18)
        add_text(c, sl, gx0 + 0.15, y + 0.06, gx1 - gx0 - 0.3, 0.3,
                 [P(s["loop_label"], size=15, bold=True, color=MINER)], "loop label", anchor="m")
    shapes = []
    for i, t in enumerate(steps):
        acc = t.get("accent")
        shp = add_card(c, sl, box.x + i * (sw + g), sy, sw, sh, step_paras[i], f"step {i + 1} {t['label']}",
                       fill=MINER if acc else WHITE, line=None if acc else MINER, anchor="m", ins=ins, radius=0.14)
        shapes.append(shp)
    for i in range(n - 1):
        cn = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, I(1), I(1))
        cn.name = f"arrow {i + 1} to {i + 2}"
        drop_style(cn)
        cn.begin_connect(shapes[i], 3)
        cn.end_connect(shapes[i + 1], 1)
        cn.line.color.rgb = rgb(SHUTTLE)
        cn.line.width = Pt(2)
        etree.SubElement(cn.line._get_or_add_ln(), qn("a:tailEnd")).attrib.update({"type": "triangle", "w": "med", "len": "med"})
    if len(loop) > 1:
        xa = box.x + loop[0] * (sw + g) + sw / 2
        xb = box.x + loop[-1] * (sw + g) + sw / 2
        yb = sy + sh
        yl = yb + ret_h * 0.62
        fb = sl.shapes.build_freeform(I(xb), I(yb))
        fb.add_line_segments([(I(xb), I(yl)), (I(xa), I(yl)), (I(xa), I(yb + 0.02))], close=False)
        ret = fb.convert_to_shape()
        ret.name = "loop return arrow"
        drop_style(ret)
        ret.fill.background()
        ret.line.color.rgb = rgb(MINER)
        ret.line.width = Pt(2.25)
        etree.SubElement(ret.line._get_or_add_ln(), qn("a:tailEnd")).attrib.update({"type": "triangle", "w": "med", "len": "med"})
    y += gh + gap
    for i, v in enumerate(s["variants"]):
        add_card(c, sl, box.x + i * (vw + 0.3), y, vw, vh, var_paras[i], f"variant {v['name']}", anchor="m")
    y += vh + gap
    add_card(c, sl, box.x, y, box.w, chh, call, "callout", fill=TANGO_TINT, anchor="m", ins=(0.3, 0.12, 0.3, 0.12))


# --------------------------------------------------------------------------- slide types: charts

def style_chart_text(c, chart, size=14):
    chart.font.size = Pt(size)
    chart.font.name = c.face["body"]
    chart.font.color.rgb = rgb(INK)


def style_value_axis(chart, lo, hi, unit, fmt):
    va = chart.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = lo, hi, unit
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(PEBBLE)
    va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.major_tick_mark = XL_TICK_MARK.NONE
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.size = Pt(13)
    va.tick_labels.font.color.rgb = rgb(SHUTTLE)


def style_category_axis(chart, size=14):
    ca = chart.category_axis
    ca.format.line.color.rgb = rgb(SHUTTLE)
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.has_major_gridlines = False
    ca.tick_labels.font.size = Pt(size)


def nice_unit(lo, hi, target=5):
    span = hi - lo
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw - 1e-12:
            return m * mag
    return raw


def add_reference_series(chart, color, dash="dash"):
    """Move the chart's second bar series into a line chart sharing the axes (an exact horizontal reference)."""
    plot_area = chart._chartSpace.chart.plotArea
    bar = plot_area.find(qn("c:barChart"))
    ser = bar.findall(qn("c:ser"))[1]
    bar.remove(ser)
    for tag in ("c:invertIfNegative", "c:spPr", "c:dPt", "c:dLbls"):
        for el in ser.findall(qn(tag)):
            ser.remove(el)
    tx = ser.find(qn("c:tx"))
    sppr = etree.fromstring(
        f'<c:spPr xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        f'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:ln w="22225" cap="flat">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:prstDash val="{dash}"/></a:ln></c:spPr>')
    marker = etree.fromstring('<c:marker xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:symbol val="none"/></c:marker>')
    tx.addnext(sppr)
    sppr.addnext(marker)
    etree.SubElement(ser, qn("c:smooth")).set("val", "0")
    lc = etree.Element(qn("c:lineChart"))
    etree.SubElement(lc, qn("c:grouping")).set("val", "standard")
    etree.SubElement(lc, qn("c:varyColors")).set("val", "0")
    lc.append(ser)
    etree.SubElement(lc, qn("c:marker")).set("val", "1")
    for ax in bar.findall(qn("c:axId")):
        etree.SubElement(lc, qn("c:axId")).set("val", ax.get("val"))
    bar.addnext(lc)


def chart_frame(c, sl, s, box, side_ratio, caption_extra=""):
    """Caption above the chart on the left, side cards on the right; returns (chart Box, side Box)."""
    ch = s["chart"]
    side_w = side_ratio * box.w
    items = s["side"]
    for extra in (0.0, 0.5, 1.0):
        sw = side_w + extra * box.w / 12.33
        if side_fit_step(items, sw, box.h, 0.18) is not None:
            break
    side_w = sw
    chart_w = box.w - side_w - 0.35
    cap = [P(ch["caption"] + caption_extra, size=13, color=SHUTTLE)]
    cap_h = text_height(cap, chart_w) + 0.1
    add_text(c, sl, box.x, box.y, chart_w, cap_h, cap, "chart caption")
    cbox = Box(box.x, box.y + cap_h, chart_w, box.h - cap_h)
    side_column(c, sl, box.r - side_w, box.y, side_w, box.h, items)
    return cbox


def build_bar_chart(c, sl, s, box):
    ch = s["chart"]
    ref = ch.get("reference_line")
    extra = f" \u00b7 dashed: {ref['label']}" if ref else ""
    cb = chart_frame(c, sl, s, box, 0.36, extra)
    cd = CategoryChartData()
    cd.categories = ch["categories"]
    cd.add_series("Final ACC", ch["values"])
    if ref:
        cd.add_series(ref["label"], [ref["value"]] * len(ch["categories"]))
    gf = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, I(cb.x), I(cb.y), I(cb.w), I(cb.h), cd)
    gf.name = "bar chart"
    chart = gf.chart
    chart.has_legend = False
    chart.has_title = False
    style_chart_text(c, chart)
    plot = chart.plots[0]
    plot.gap_width = 55
    plot.vary_by_categories = False
    ser = plot.series[0]
    for i, cat in enumerate(ch["categories"]):
        pt = ser.points[i]
        pt.format.fill.solid()
        col = TANGO if cat == ch.get("highlight") else series_color(c, cat)
        pt.format.fill.fore_color.rgb = rgb(col)
    labels = ch.get("labels")
    dec = len(labels[0].split(".")[1]) if labels and "." in labels[0] else 3
    if labels and any(lbl != f"{v:.{dec}f}" for lbl, v in zip(labels, ch["values"])):
        c.warn("bar labels differ from the formatted values; showing formatted values")
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = "0." + "0" * dec
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(15)
    dl.font.bold = True
    lo, hi = ch["axis_min"], ch["axis_max"]
    style_value_axis(chart, lo, hi, 0.02 if hi - lo <= 0.2 else nice_unit(lo, hi), "0.00")
    style_category_axis(chart, 15)
    if ref:
        add_reference_series(chart, SHUTTLE)
    return sl


def build_line_chart(c, sl, s, box):
    ch = s["chart"]
    cb = chart_frame(c, sl, s, box, 0.37)
    cd = CategoryChartData()
    cd.categories = [str(x) for x in ch["x"]]
    for ser in ch["series"]:
        cd.add_series(ser["name"], ser["values"])
    gf = sl.shapes.add_chart(XL_CHART_TYPE.LINE, I(cb.x), I(cb.y), I(cb.w), I(cb.h), cd)
    gf.name = "line chart"
    chart = gf.chart
    chart.has_title = False
    style_chart_text(c, chart, 13)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(14)
    for ser, spec in zip(chart.plots[0].series, ch["series"]):
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.NONE
        ser.format.line.color.rgb = rgb(series_color(c, spec["name"]))
        ser.format.line.width = Pt(2.5)
    lo, hi = ch["axis_min"], ch["axis_max"]
    style_value_axis(chart, lo, hi, 0.35 if abs(hi - lo - 1.05) < 1e-9 else nice_unit(lo, hi), "0.00")
    style_category_axis(chart, 12)
    ca = chart.category_axis
    ca.has_title = True
    ca.axis_title.text_frame.text = ch["x_label"]
    run = ca.axis_title.text_frame.paragraphs[0].runs[0]
    run.font.size = Pt(13)
    run.font.bold = False
    run.font.color.rgb = rgb(SHUTTLE)
    return sl


def build_clustered_bar(c, sl, s, box):
    ch = s["chart"]
    cb = chart_frame(c, sl, s, box, 0.34)
    cd = CategoryChartData()
    cd.categories = ch["categories"]
    for ser in ch["series"]:
        cd.add_series(ser["name"], ser["values"])
    gf = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, I(cb.x), I(cb.y), I(cb.w), I(cb.h), cd)
    gf.name = "clustered bar chart"
    chart = gf.chart
    chart.has_title = False
    style_chart_text(c, chart, 14)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(14)
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.overlap = -5
    for ser, col in zip(plot.series, (MINER, LIMA)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(col)
    fmt = "0%" if ch.get("percent") else "0.0"
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(14)
    dl.font.bold = True
    top = max(max(x["values"]) for x in ch["series"])
    hi = math.ceil(top * 10) / 10
    style_value_axis(chart, 0, hi, 0.2 if hi <= 0.8 else nice_unit(0, hi), fmt)
    style_category_axis(chart, 15)
    return sl


# --------------------------------------------------------------------------- slide type: forest plot

def fnum(v, dec, sign=True):
    """Signed fixed-point text; rounds half away from zero on the decimal value (not the binary float)."""
    q = Decimal(str(v)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    if q == 0:
        return f"{abs(q):.{dec}f}"
    return ("\u2212" if q < 0 else "+" if sign else "") + f"{abs(q):.{dec}f}"


def row_decimals(row):
    vals = (row["mean"], row["lo"], row["hi"])
    has4 = any(abs(round(v * 1000) - v * 1000) > 1e-6 for v in vals)
    return 4 if has4 and any(abs(v) < 0.01 for v in vals) else 3


def value_text(row, dec=None):
    d = dec or row_decimals(row)
    return f"{fnum(row['mean'], d)} [{fnum(row['lo'], d)}, {fnum(row['hi'], d)}]"


def tick_values(lo, hi):
    for step in (0.01, 0.02, 0.05, 0.1, 0.2):
        k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
        if k1 - k0 + 1 <= 7:
            return [round(k * step, 6) for k in range(k0, k1 + 1)], step
    return [lo, hi], hi - lo


def tick_label(v, fmt):
    if abs(v) < 1e-9:
        return "0"
    dec = len(fmt.split(".")[1]) if "." in fmt else 0
    return fnum(v, dec, sign=fmt.startswith("+"))


class ForestPlan:
    """Geometry of one forest plot inside a region (caption, panels, rows, value column)."""

    def __init__(self, c, ch, region, label_pt, value_pt):
        self.c, self.ch, self.region = c, ch, region
        self.panels = ch["panels"]
        rows = [r for p in self.panels for r in p["rows"]]
        self.nrows = len(rows)
        self.has_note = any(r.get("note") for r in rows)
        self.ref = ch.get("reference_line")
        self.label_pt, self.value_pt = label_pt, value_pt
        lab = lambda r: P(r["label"], size=label_pt, bold=bool(r.get("emphasis")))
        lab_need = max(max_line_width(lab(r)) for r in rows) + 0.2
        self.label_w = min(max(lab_need, 0.9), 0.32 * region.w)
        self.dec = ch.get("decimals")
        self.value_w = max(max_line_width(P(value_text(r, self.dec), size=value_pt, bold=True)) for r in rows) + 0.15
        self.note_w = 0.95 if self.has_note else 0.0
        self.plot_w = region.w - self.label_w - 0.2 - self.value_w - self.note_w - 0.1
        cap = [P(ch["caption"], size=13, color=SHUTTLE)]
        self.cap_h = text_height(cap, region.w) + 0.12
        self.top_pad = 0.34 if (self.ref or self.has_note) else 0.0
        self.panel_head = 0.3
        self.tick_h = 0.27
        self.panel_gap = 0.1
        fixed = self.cap_h + self.top_pad
        for p in self.panels:
            fixed += (self.panel_head if p["name"] else 0) + self.tick_h + self.panel_gap
        self.needs = [[max(0.3, para_lines(lab(r), self.label_w - 0.1) * label_pt * LINE / 72 + 0.06) for r in p["rows"]]
                      for p in self.panels]
        self.extra = (region.h - fixed - sum(map(sum, self.needs))) / self.nrows
        self.feasible = self.extra >= 0 and self.plot_w >= 2.4
        pad = min(0.3, max(0.0, self.extra))
        self.row_h = [[n + pad for n in ns] for ns in self.needs]


def plan_forest(c, ch, region, side):
    """Pick fonts and where the side cards go; returns (plan, mode, card_step)."""
    fonts = ((14, 14), (14, 13), (13, 13))
    plan = None
    if not side:
        for lp, vp in fonts:
            plan = ForestPlan(c, ch, region, lp, vp)
            if plan.feasible:
                break
        return plan, "none", 0
    sw = min(0.33 * region.w, 4.2)
    for lp, vp in fonts:
        for step in range(4):
            row_h = side_row(c, None, 0, 0, region.w, side, step)
            plan = ForestPlan(c, ch, Box(region.x, region.y, region.w, region.h - row_h - 0.25), lp, vp)
            if plan.feasible and plan.extra >= 0.1:
                return plan, "below", step
        if side_fit_step(side, sw, region.h, 0.18) is not None:
            right = ForestPlan(c, ch, Box(region.x, region.y, region.w - sw - 0.3, region.h), lp, vp)
            if right.feasible:
                return right, "right", 0
    return plan, "below", 3


def build_forest(c, sl, s, box):
    ch = s["chart"]
    y = box.y
    if s.get("intro"):
        paras, _, _ = fit(lambda z: [P(s["intro"], size=z)], box.w, 0.9, [16, 15, 14], c, "intro")
        h = text_height(paras, box.w) + 0.04
        add_text(c, sl, box.x, y, box.w, h, paras, "intro")
        y += h + 0.16
    if s.get("quotes"):
        qs = s["quotes"]
        gap = 0.3
        qw = (box.w - gap) / 2
        for z in (16, 15, 14):
            qp = [[P(q["label"], size=z - 1, bold=True, color=MINER, after=3), P(q["text"], size=z, italic=True)] for q in qs]
            qh = max(card_need(p, qw) for p in qp)
            if qh <= 1.75:
                break
        for i, q in enumerate(qs):
            add_card(c, sl, box.x + i * (qw + gap), y, qw, qh, qp[i], f"quote card {q['label']}", anchor="m")
        y += qh + 0.2
    region = Box(box.x, y, box.w, box.b - y)
    side = s.get("side") or []
    plan, mode, card_step = plan_forest(c, ch, region, side)
    if not plan.feasible:
        c.warn("forest plot rows are cramped (spare row height %.2f in)" % plan.extra)
    r = plan.region
    # --- caption
    add_text(c, sl, r.x, r.y, r.w, plan.cap_h, [P(ch["caption"], size=13, color=SHUTTLE)], "plot caption")
    plot_x = r.x + plan.label_w + 0.2
    pw = plan.plot_w
    xmin, xmax = ch["x_min"], ch["x_max"]
    px = lambda v: plot_x + (v - xmin) / (xmax - xmin) * pw
    ticks, _ = tick_values(xmin, xmax)
    fmt = ch.get("x_format", "+0.00")
    yy = r.y + plan.cap_h + plan.top_pad
    first_top = yy
    row_tops = []
    for pi, panel in enumerate(plan.panels):
        if panel["name"]:
            add_text(c, sl, r.x, yy, r.w, plan.panel_head, [P(panel["name"], size=14, bold=True, color=FOREST)],
                     f"panel heading {pi + 1}", anchor="m")
            yy += plan.panel_head
        top = yy
        heights = plan.row_h[pi]
        bottom = top + sum(heights)
        for t in ticks:
            if abs(t) < 1e-9:
                add_line(sl, px(0), top - 0.03, px(0), bottom + 0.03, SHUTTLE, 1.25, f"zero line {pi + 1}")
            else:
                add_line(sl, px(t), top - 0.03, px(t), bottom + 0.03, PEBBLE, 0.75, f"tick line {t:+g} panel {pi + 1}")
            add_text(c, sl, px(t) - 0.4, bottom + 0.05, 0.8, 0.24, [P(tick_label(t, fmt), size=12, color=SHUTTLE, align="c")],
                     f"tick label {t:+g} panel {pi + 1}", anchor="t")
        for ri, row in enumerate(panel["rows"]):
            ry = top + sum(heights[:ri])
            rh = heights[ri]
            cy = ry + rh / 2
            emph = bool(row.get("emphasis"))
            col = TANGO if emph else series_color(c, row["label"])
            add_text(c, sl, r.x, ry, plan.label_w, rh, [P(row["label"], size=plan.label_pt, bold=emph, color=INK)],
                     f"row label {row['label'][:30]}", anchor="m")
            add_line(sl, px(row["lo"]), cy, px(row["hi"]), cy, col, 2.25, f"interval {row['label'][:30]}")
            d = 0.17
            add_box(sl, MSO_SHAPE.OVAL, px(row["mean"]) - d / 2, cy - d / 2, d, d, f"mean {row['label'][:30]}", fill=col)
            vx = plot_x + pw + 0.15
            add_text(c, sl, vx, ry, plan.value_w, rh, [P(value_text(row, plan.dec), size=plan.value_pt, bold=emph, color=INK)],
                     f"value {row['label'][:30]}", anchor="m")
            if row.get("note"):
                add_text(c, sl, vx + plan.value_w, ry, plan.note_w, rh, [P(row["note"], size=plan.value_pt, color=SHUTTLE)],
                         f"seeds above zero {row['label'][:30]}", anchor="m")
        row_tops.append((top, bottom))
        yy = bottom + plan.tick_h + plan.panel_gap
    if plan.has_note:
        add_text(c, sl, plot_x + pw + 0.15 + plan.value_w, first_top - 0.32, plan.note_w, 0.28,
                 [P("seeds > 0", size=12, color=SHUTTLE)], "notes header", anchor="b")
    if plan.ref:
        rv = plan.ref["value"]
        top0, bot_last = row_tops[0][0], row_tops[-1][1]
        add_line(sl, px(rv), top0 - 0.03, px(rv), bot_last + 0.03, ORIENT, 1.75, "reference line", dash=MSO_LINE.DASH)
        lab = [P(plan.ref["label"], size=12, color=ORIENT, bold=True)]
        lw = max_line_width(lab[0]) + 0.1
        right_limit = (plot_x + pw + 0.15 + plan.value_w - 0.1) if plan.has_note else r.r
        lx = min(max(px(rv) - lw / 2, plot_x - 0.2), right_limit - lw)
        add_text(c, sl, lx, top0 - 0.32, lw, 0.27, lab, "reference line label", anchor="b")
    # --- side cards
    if mode == "right":
        sw = min(0.33 * region.w, 4.2)
        side_column(c, sl, region.r - sw, region.y, sw, region.h, side)
    elif mode == "below":
        row_h = side_row(c, None, 0, 0, region.w, side, card_step)
        side_row(c, sl, region.x, region.b - row_h, region.w, side, card_step)
    return sl


# --------------------------------------------------------------------------- slide type: equation

def build_equation(c, sl, s, box):
    eqs, notes, stats = s["equations"], s["equation_notes"], s["stats"]
    gap = 0.3
    chosen = None
    for ez, nz, (a, b) in ((26, 18, (36, 17)), (25, 17, (34, 16)), (24, 16, (32, 16)), (24, 15, (30, 15)), (24, 14, (28, 14))):
        eq_paras = [P(e, size=ez, font="eq", align="c", eq=True, color=INK, line=ez * 1.5, after=ez * 0.25) for e in eqs]
        eh = card_need(eq_paras, box.w, (0.3, 0.2, 0.3, 0.2))
        note_paras = [P(t, size=nz, align="c", color=INK, after=4) for t in notes]
        nh = text_height(note_paras, box.w * 0.8) + 0.05
        cw = (box.w - gap) / 2
        st_paras = [[P(stat_text(x["stat"]), size=a, font="head", color=FOREST, eq=True, align="c", after=4),
                     P(x["text"], size=b, align="c")] for x in stats]
        sh = max(card_need(p, cw) for p in st_paras)
        chosen = (eq_paras, eh, note_paras, nh, st_paras, sh, cw)
        if eh + nh + sh + 2 * gap <= box.h:
            break
    else:
        c.warn("equation slide does not fit at the minimum sizes")
    eq_paras, eh, note_paras, nh, st_paras, sh, cw = chosen
    extra = max(0.0, (box.h - (eh + nh + sh + 2 * gap)) / 3)
    y = box.y
    add_card(c, sl, box.x, y, box.w, eh + extra, eq_paras, "equations", anchor="m", ins=(0.3, 0.2, 0.3, 0.2))
    y += eh + extra + gap
    add_text(c, sl, box.x + box.w * 0.1, y, box.w * 0.8, nh + extra, note_paras, "equation notes", anchor="m")
    y += nh + extra + gap
    for i, p in enumerate(st_paras):
        add_card(c, sl, box.x + i * (cw + gap), y, cw, sh + extra, p, f"stat card {i + 1}", anchor="m")


# --------------------------------------------------------------------------- slide types: tables

def set_cell_borders(cell, top, color=PEBBLE, w_pt=0.75):
    """Hairline rules above (if top) and below the cell; these must precede the fill in tcPr."""
    tcpr = cell._tc.get_or_add_tcPr()
    for pos, tag in enumerate(["a:lnT", "a:lnB"] if top else ["a:lnB"]):
        ln = etree.Element(qn(tag), w=str(int(w_pt * 12700)), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
        etree.SubElement(ln, qn("a:prstDash")).set("val", "solid")
        tcpr.insert(pos, ln)


def add_table(c, sl, x, y, col_w, rows, name, min_row=0.36, header_fill=FOREST, size=16, fill_h=None):
    """rows: list of rows of dicts {text, bold, color, fill, align, size}; first row is the header."""
    ncols = len(col_w)
    mar = 0.1
    heights = []
    for row in rows:
        hh = 0.0
        for j, cell in enumerate(row):
            p = P(cell["text"], size=cell.get("size", size), bold=cell.get("bold", False))
            hh = max(hh, para_lines(p, col_w[j] - 2 * mar) * cell.get("size", size) * LINE / 72 + 0.1)
        heights.append(max(min_row, hh))
    if fill_h and fill_h > sum(heights):
        grow = (fill_h - sum(heights)) / len(heights)
        heights = [h + grow for h in heights]
    gf = sl.shapes.add_table(len(rows), ncols, I(x), I(y), I(sum(col_w)), I(sum(heights)))
    gf.name = name
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, w in enumerate(col_w):
        tbl.columns[j].width = I(w)
    for i, row in enumerate(rows):
        tbl.rows[i].height = I(heights[i])
        for j, cell in enumerate(row):
            tc = tbl.cell(i, j)
            tc.margin_left = tc.margin_right = I(mar)
            tc.margin_top = tc.margin_bottom = I(0.04)
            tc.vertical_anchor = MSO_ANCHOR.MIDDLE
            tc.fill.solid()
            tc.fill.fore_color.rgb = rgb(cell.get("fill", WHITE))
            tf = tc.text_frame
            tf.word_wrap = True
            p = P(cell["text"], size=cell.get("size", size), bold=cell.get("bold", False), color=cell.get("color", INK),
                  align=cell.get("align", "l"))
            write_para(c, tf.paragraphs[0], p)
            set_cell_borders(tc, top=i > 1)
    return sum(heights)


def build_stat_table(c, sl, s, box):
    t = s["table"]
    cols = t["columns"]
    hi = cols.index(t["highlight_column"]) if t.get("highlight_column") in cols else -1
    first_w = box.w * 0.52
    other_w = (box.w - first_w) / (len(cols) - 1)
    col_w = [first_w] + [other_w] * (len(cols) - 1)
    cap = [P(t["caption"], size=13, color=SHUTTLE)]
    cap_h = text_height(cap, box.w) + 0.12
    add_text(c, sl, box.x, box.y, box.w, cap_h, cap, "table caption")
    callout = [P(s["callout"], size=19, bold=True, color=FOREST)]
    call_h = card_need(callout, box.w, (0.3, 0.14, 0.3, 0.14))
    rows = [[{"text": h, "bold": True, "color": WHITE, "fill": FOREST, "align": "l" if j == 0 else "r"} for j, h in enumerate(cols)]]
    for r in t["rows"]:
        row = []
        for j, v in enumerate(r):
            row.append({"text": v, "align": "l" if j == 0 else "r", "bold": j == hi, "fill": TANGO_TINT if j == hi else WHITE})
        rows.append(row)
    for size in (18, 17, 16, 15, 14):
        for rr in rows:
            for cell in rr:
                cell["size"] = size
        avail = box.h - cap_h - call_h - 0.5
        need = sum(max(0.5, max(para_lines(P(cell["text"], size=size, bold=cell["bold"] if "bold" in cell else False), col_w[j] - 0.2)
                                * size * LINE / 72 + 0.1 for j, cell in enumerate(rr))) for rr in rows)
        if need <= avail:
            break
    else:
        c.warn("table does not fit at 14 pt")
    th = add_table(c, sl, box.x, box.y + cap_h, col_w, rows, "stat table", min_row=min(0.7, (box.h - cap_h - call_h - 0.6) / len(rows)), size=size)
    cy = box.y + cap_h + th + min(0.4, max(0.25, (box.h - cap_h - th - call_h) / 2))
    add_card(c, sl, box.x, cy, box.w, call_h, callout, "callout", fill=TANGO_TINT, anchor="m", ins=(0.3, 0.14, 0.3, 0.14))


def build_claims_table(c, sl, s, box):
    t = s["table"]
    cols = t["columns"]
    ratios = [0.52, 0.235, 0.155, 0.09]
    col_w = [box.w * r for r in ratios]
    col_w[-1] = box.w - sum(col_w[:-1])
    bottom = [P(s["bottom_line"], size=18, bold=True, color=FOREST)]
    bl_h = text_height(bottom, box.w) + 0.05
    for size in (14, 13):
        rows = [[{"text": h, "bold": True, "color": WHITE, "fill": FOREST, "size": size} for h in cols]]
        for r in t["rows"]:
            status = r[3]
            rows.append([{"text": r[0], "size": size}, {"text": r[1], "size": size, "color": SHUTTLE},
                         {"text": r[2], "size": size, "color": SHUTTLE},
                         {"text": status, "size": size, "bold": True, "color": MINER if status == "shown" else RUBY}])
        need = 0.0
        for rr in rows:
            need += max(0.34, max(para_lines(P(cell["text"], size=size, bold=cell.get("bold", False)), col_w[j] - 0.2)
                                  * size * LINE / 72 + 0.1 for j, cell in enumerate(rr)))
        if need + bl_h + 0.3 <= box.h:
            break
    else:
        c.warn("claims table does not fit at 13 pt")
    th = add_table(c, sl, box.x, box.y, col_w, rows, "claims table", size=size, fill_h=box.h - bl_h - 0.3)
    add_text(c, sl, box.x, box.y + th + 0.3, box.w, bl_h, bottom, "bottom line")


# --------------------------------------------------------------------------- driver

BUILDERS = {
    "flow": build_flow, "two_column": build_two_column, "bar_chart": build_bar_chart, "line_chart": build_line_chart,
    "clustered_bar": build_clustered_bar, "forest": build_forest, "equation": build_equation,
    "stat_table": build_stat_table, "claims_table": build_claims_table, "answer": build_answer,
    "placeholder": build_placeholder, "bullets": build_bullets,
}


def load_template(path):
    """Open .pptx or .potx; a .potx is copied with its main part's content type rewritten to a presentation."""
    if not path.lower().endswith(".potx"):
        return Presentation(path), None
    tmpdir = tempfile.mkdtemp(prefix="deck_tpl_")
    out = os.path.join(tmpdir, "template.pptx")
    old = "application/vnd.openxmlformats-officedocument.presentationml.template.main+xml"
    new = "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.decode("utf-8").replace(old, new).encode("utf-8")
            zout.writestr(item, data)
    return Presentation(out), tmpdir


def add_sections(prs, slide_parts):
    """slide_parts: list of part keys in slide order; writes one PowerPoint section per part."""
    p14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
    ids = [sld.get("id") for sld in prs.slides._sldIdLst]
    groups = []
    for sid, part in zip(ids, slide_parts):
        if groups and groups[-1][0] == part:
            groups[-1][1].append(sid)
        else:
            groups.append((part, [sid]))
    root = prs.part._element
    ext_lst = root.find(qn("p:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(root, qn("p:extLst"))
    ext = etree.SubElement(ext_lst, qn("p:ext"), uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    lst = etree.SubElement(ext, f"{{{p14}}}sectionLst", nsmap={"p14": p14})
    for part, sids in groups:
        sec = etree.SubElement(lst, f"{{{p14}}}section", name=SECTION_NAMES.get(str(part), f"Part {part}"),
                               id="{" + str(uuid.uuid4()).upper() + "}")
        sl = etree.SubElement(sec, f"{{{p14}}}sldIdLst")
        for sid in sids:
            etree.SubElement(sl, f"{{{p14}}}sldId", id=sid)


def build(args):
    with open(args.content, encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    tmpdir = None
    if args.template:
        prs, tmpdir = load_template(args.template)
        mode = "template"
        clear_slides(prs)
    else:
        prs = Presentation()
        prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
        mode = "fallback"
    c = Ctx(mode, content, prs)
    if mode == "template":
        c.layouts = pick_layouts(prs)
        print("template layouts used: title -> '%s', content -> '%s', closing -> '%s'" % (
            c.layouts["title"].name, c.layouts["content"].name, c.layouts["closing"].name))
    only = {x.strip() for x in args.only.split(",")} if args.only else None
    slides = content["slides"]
    total = len(slides)
    parts = []
    for n, s in enumerate(slides, start=1):
        if only and str(s["id"]) not in only:
            continue
        c.slide_id = str(s["id"])
        if s["type"] == "title":
            sl = build_title(c, s, n)
        elif s["type"] == "closing":
            sl = build_closing(c, s, n)
        else:
            sl, box = content_frame(c, s, n)
            BUILDERS[s["type"]](c, sl, s, box)
        write_notes(sl, s)
        parts.append(s["part"])
    add_sections(prs, parts)
    prs.core_properties.title = content["meta"]["title"]
    prs.core_properties.author = ""
    prs.core_properties.last_modified_by = ""
    prs.save(args.out)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"wrote {args.out}: {len(parts)} slides ({mode} mode)")
    for w in c.warnings:
        print("WARNING:", w)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Build the talk deck from deck.yaml")
    ap.add_argument("--content", help="deck.yaml")
    ap.add_argument("--out", help="output .pptx")
    ap.add_argument("--template", help="official template (.pptx or .potx); omit for the built-in S&T look")
    ap.add_argument("--only", help="comma-separated slide ids to build, e.g. 1.4,3.6")
    ap.add_argument("--print-layouts", action="store_true", help="list the template's layouts and placeholders, then exit")
    args = ap.parse_args()
    if args.print_layouts:
        if not args.template:
            ap.error("--print-layouts needs --template")
        prs, tmp = load_template(args.template)
        print_layouts(prs)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0
    if not args.content or not args.out:
        ap.error("--content and --out are required")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
