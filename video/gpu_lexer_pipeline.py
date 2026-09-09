from manim import *
from pathlib import Path
import manimpango


FONT_DIR = Path(__file__).parent / "assets/fonts"
FONT_FILES = [FONT_DIR / "GeistMono-Regular.otf"]
for font_file in FONT_FILES:
    if not manimpango.register_font(str(font_file.resolve())):
        raise RuntimeError(f"could not register {font_file}")
CODE_FONT = "Geist Mono"
TEXT_REFERENCE_SIZE = 48
BACKGROUND = "#000000"
WHITE_TEXT = "#F5F5F5"
MUTED = "#999999"
LINE = "#707070"
ACTIVATION = "#3B82F6"

SEMANTIC = {
    "plain": "#D4D4D4",
    "comment": "#6A9955",
    "string": "#A6E22E",
    "number": "#F78C6C",
    "keyword": "#C792EA",
    "type": "#FFCB6B",
    "function": "#82AAFF",
    "constant": "#F07178",
    "operator": "#89DDFF",
}

PARTS = [
    ("const", "word", "keyword"),
    (" ", "space", "plain"),
    ("message", "word", "plain"),
    (" ", "space", "plain"),
    ("=", "symbol", "operator"),
    (" ", "space", "plain"),
    ('"', "symbol", "string"),
    ("hello", "word", "string"),
    (",", "symbol", "string"),
    (" ", "space", "plain"),
    ("world", "word", "string"),
    ("!", "symbol", "string"),
    ('"', "symbol", "string"),
    (";", "symbol", "plain"),
    ("\n", "newline", "plain"),
    ("console", "word", "plain"),
    (".", "symbol", "plain"),
    ("log", "word", "function"),
    ("(", "symbol", "plain"),
    ("message", "word", "plain"),
    (")", "symbol", "plain"),
]

SHORT_LABEL = {
    "plain": "plain",
    "string": "str",
    "keyword": "keyword",
    "function": "func",
    "operator": "op",
}


def geist_mono(text, size=24, color=WHITE_TEXT, **kwargs):
    # Pango snaps small font sizes to coarse glyph advances: Geist Mono jumps
    # by ~46% between sizes 14 and 15. Shape once at a stable size and scale the
    # vectors so requested sizes preserve exactly proportional tracking.
    return Text(
        text,
        font=CODE_FONT,
        font_size=TEXT_REFERENCE_SIZE,
        color=color,
        disable_ligatures=False,
        **kwargs,
    ).scale(size / TEXT_REFERENCE_SIZE)


def caption(text):
    return geist_mono(text, 28).move_to(UP * 3.5)


def note(text, y=-3.45, color=MUTED):
    return geist_mono(text, 17, color).move_to(UP * y)


def baseline_glyph(text, size, color):
    # Prefix each isolated part with the same cap/descender pair, then retain
    # only the requested glyphs. Pango lays every retained glyph on one shared
    # font baseline instead of vertically centering punctuation independently.
    layout = geist_mono(f"Ag{text}", size, color)
    layout.move_to(ORIGIN)
    glyph = VGroup(*layout[2:])
    glyph.shift(LEFT * glyph.get_x())
    return glyph


def visible_char(character):
    if character == " ":
        return "␣"
    if character == "\n":
        return "↵"
    return character


def stable_values(key, count=5):
    state = 2166136261
    for character in key:
        state = ((state ^ ord(character)) * 16777619) & 0xFFFFFFFF
    values = []
    for index in range(count):
        state = (1664525 * (state ^ index) + 1013904223) & 0xFFFFFFFF
        values.append(0.34 + 0.64 * ((state >> 8) & 255) / 255)
    return values


class CharacterCell(VGroup):
    def __init__(self, character):
        box = RoundedRectangle(
            width=0.33,
            height=0.45,
            corner_radius=0.1,
            stroke_color=LINE,
            stroke_width=1.2,
            fill_color="#090909",
            fill_opacity=1,
        )
        glyph = baseline_glyph(
            visible_char(character),
            14,
            MUTED if character == " " else WHITE_TEXT,
        )
        super().__init__(box, glyph)


class PartCell(VGroup):
    def __init__(self, source, kind, semantic):
        visible = visible_char(source)
        if kind in {"space", "newline", "symbol"}:
            width = 0.36
        else:
            width = 0.34 + len(visible) * 0.115
        self.box = RoundedRectangle(
            width=width,
            height=0.55,
            corner_radius=0.15,
            stroke_color=LINE,
            stroke_width=1.2,
            fill_color="#090909",
            fill_opacity=1,
        )
        self.glyph = baseline_glyph(
            visible,
            17,
            MUTED if kind in {"space", "newline"} else WHITE_TEXT,
        )
        self.source = source
        self.kind = kind
        self.semantic = semantic
        super().__init__(self.box, self.glyph)


def make_code_block(color=WHITE_TEXT):
    first = geist_mono('const message = "hello, world!";', 39, color)
    second = geist_mono("console.log(message)", 39, color)
    return VGroup(first, second).arrange(DOWN, aligned_edge=LEFT, buff=0.32)


def make_character_grid():
    first_source = 'const message = "hello, world!";\n'
    second_source = "console.log(message)"
    first = VGroup(*[CharacterCell(character) for character in first_source]).arrange(RIGHT, buff=0.025)
    second = VGroup(*[CharacterCell(character) for character in second_source]).arrange(RIGHT, buff=0.025)
    grid = VGroup(first, second).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    if grid.width > 12.8:
        grid.scale_to_fit_width(12.8)
    return grid


def make_part_strip():
    cells = VGroup(*[PartCell(*part) for part in PARTS]).arrange(RIGHT, buff=0.045)
    if cells.width > 12.85:
        cells.scale_to_fit_width(12.85)
    return cells


def activation_strip(key, center, width=0.12, height=0.78, cells=6):
    values = stable_values(key, cells)
    gap = 0.018
    cell_height = (height - gap * (cells - 1)) / cells
    items = VGroup()
    for value in values:
        item = RoundedRectangle(
            width=width,
            height=cell_height,
            corner_radius=min(0.02, cell_height / 4),
            stroke_width=0,
            fill_color=ACTIVATION,
            fill_opacity=value,
        )
        items.add(item)
    items.arrange(UP, buff=gap).move_to(center)
    return items


def activation_node(key, center, width=0.3, height=0.13, cells=4):
    values = stable_values(key, cells)
    gap = 0.012
    cell_width = (width - gap * (cells - 1)) / cells
    items = VGroup()
    for value in values:
        item = RoundedRectangle(
            width=cell_width,
            height=height,
            corner_radius=0.015,
            stroke_width=0.45,
            stroke_color="#6BA2FF",
            fill_color=ACTIVATION,
            fill_opacity=value,
        )
        items.add(item)
    items.arrange(RIGHT, buff=gap).move_to(center)
    return items


def make_tree(xs, leaf_y=-1.43, step=0.69):
    leaf_nodes = VGroup(*[
        activation_node(PARTS[index][0], [x, leaf_y, 0])
        for index, x in enumerate(xs)
    ])
    levels = [leaf_nodes]
    edge_levels = []
    centers = [node.get_center() for node in leaf_nodes]
    level_index = 1
    while len(centers) > 1:
        parent_centers = []
        parent_nodes = VGroup()
        edges = VGroup()
        for index in range(0, len(centers), 2):
            children = centers[index:index + 2]
            x = sum(point[0] for point in children) / len(children)
            center = [x, leaf_y + level_index * step, 0]
            parent = activation_node(f"{level_index}:{index}", center)
            parent_nodes.add(parent)
            for child_index, child in enumerate(children):
                child_node = levels[-1][index + child_index]
                edges.add(Line(
                    child_node.get_top(),
                    parent.get_bottom(),
                    color=LINE,
                    stroke_width=1.15,
                ))
            parent_centers.append(parent.get_center())
        levels.append(parent_nodes)
        edge_levels.append(edges)
        centers = parent_centers
        level_index += 1
    return levels, edge_levels


def make_network_panel():
    left = VGroup(
        activation_node("leaf", [-4.6, 0.9, 0], width=0.72, height=0.16, cells=8),
        activation_node("context", [-4.6, 0.45, 0], width=0.72, height=0.16, cells=8),
    )
    left_label = geist_mono("leaf + context", 14, MUTED).next_to(left, DOWN, buff=0.3)

    hidden = VGroup(*[
        Dot(
            radius=0.055,
            color=interpolate_color(BLACK, ManimColor(ACTIVATION), 0.3 + (index % 5) * 0.14),
            fill_opacity=1,
            stroke_width=0.8,
        )
        for index in range(16)
    ]).arrange_in_grid(rows=4, cols=4, buff=(0.25, 0.2)).move_to(LEFT * 1.65 + UP * 0.65)
    hidden_label = geist_mono("72 hidden", 14, MUTED).next_to(hidden, DOWN, buff=0.3)

    input_edges = VGroup(*[
        Line(source.get_right(), target.get_left(), color=LINE, stroke_width=0.65)
        for source in left for target in hidden
    ])

    names = ["plain", "comment", "string", "number", "keyword", "type", "function", "constant", "operator"]
    scores = [0.42, 0.17, 0.28, 0.12, 0.92, 0.21, 0.36, 0.14, 0.49]
    rows = VGroup()
    for name, score in zip(names, scores):
        track = RoundedRectangle(
            width=1.7,
            height=0.15,
            corner_radius=0.05,
            stroke_color=LINE,
            stroke_width=0.8,
            fill_opacity=0,
        )
        fill = RoundedRectangle(
            width=max(0.08, 1.64 * score),
            height=0.1,
            corner_radius=0.025,
            stroke_width=0,
            fill_color=SEMANTIC[name] if name == "keyword" else MUTED,
            fill_opacity=1,
        ).move_to(track).align_to(track, LEFT).shift(RIGHT * 0.03)
        bar = VGroup(track, fill)
        label = geist_mono(name, 14, SEMANTIC[name])
        row = VGroup(bar, label).arrange(RIGHT, buff=0.2)
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.115).move_to(RIGHT * 3.55 + UP * 0.5)
    score_label = geist_mono("9 predicted types", 14, MUTED).next_to(rows, DOWN, buff=0.3)

    output_edges = VGroup()
    for hidden_dot in [hidden[0], hidden[5], hidden[10], hidden[15]]:
        for row in rows:
            output_edges.add(Line(hidden_dot.get_right(), row[0].get_left(), color=LINE, stroke_width=0.6))

    panel = VGroup(input_edges, output_edges, left, left_label, hidden, hidden_label, rows, score_label)
    panel.shift((-0.14 - panel.get_y()) * UP)
    return panel, rows


def make_type_legend(names):
    entries = VGroup()
    for name in names:
        dot = Dot(radius=0.045, color=SEMANTIC[name])
        label = geist_mono(name, 14, SEMANTIC[name])
        entries.add(VGroup(dot, label).arrange(RIGHT, buff=0.1))
    return entries.arrange(RIGHT, buff=0.42)


def make_badges(parts):
    badges = VGroup()
    for cell in parts:
        if cell.kind in {"space", "newline"}:
            continue
        label = geist_mono(SHORT_LABEL[cell.semantic], 10, SEMANTIC[cell.semantic])
        if label.width > cell.width * 0.9:
            label.scale_to_fit_width(cell.width * 0.9)
        label.next_to(cell, UP, buff=0.1)
        badges.add(label)
    return badges


def make_highlighted_code():
    # Each source line is one Pango layout. Coloring substrings after layout
    # preserves the font's spacing instead of concatenating separately measured
    # token objects.
    first = geist_mono(
        'const message = "hello, world!";',
        39,
        t2c={
            "const": SEMANTIC["keyword"],
            "=": SEMANTIC["operator"],
            '"hello, world!"': SEMANTIC["string"],
        },
    )
    second = geist_mono(
        "console.log(message)",
        39,
        t2c={"log": SEMANTIC["function"]},
    )
    return VGroup(first, second).arrange(DOWN, aligned_edge=LEFT, buff=0.32)


class GpuLexerPipeline(Scene):
    slowdown = 1.55

    def play(self, *animations, **kwargs):
        kwargs["run_time"] = kwargs.get("run_time", 1) * self.slowdown
        kwargs.setdefault("rate_func", smooth)
        return super().play(*animations, **kwargs)

    def wait(self, duration=1, **kwargs):
        return super().wait(duration * self.slowdown, **kwargs)

    def setup(self):
        self.camera.background_color = BACKGROUND

    def construct(self):
        # The unlabeled source string.
        current_caption = caption("Source code")
        source = make_code_block().move_to(UP * 0.2)
        self.play(
            FadeIn(current_caption, shift=DOWN * 0.08),
            LaggedStart(*[AddTextLetterByLetter(line) for line in source], lag_ratio=0.34),
            run_time=1.35,
        )
        no_language = note("no language id", y=-2.05)
        self.play(FadeIn(no_language, shift=UP * 0.08), run_time=0.45)
        self.wait(0.45)

        # Explanatory characters, then the actual simple parts.
        next_caption = caption("Characters")
        characters = make_character_grid().move_to(UP * 0.05)
        self.play(
            ReplacementTransform(current_caption, next_caption),
            ReplacementTransform(source, characters),
            FadeOut(no_language),
            run_time=0.9,
        )
        current_caption = next_caption
        self.wait(0.3)

        parts = make_part_strip().move_to(DOWN * 0.15)
        next_caption = caption("One CPU scan → simple parts")
        part_note = note("words · spaces · newlines · symbols", y=-1.45)
        self.play(
            ReplacementTransform(current_caption, next_caption),
            ReplacementTransform(characters, parts),
            FadeIn(part_note, shift=UP * 0.08),
            run_time=1.05,
        )
        current_caption = next_caption
        self.wait(0.55)

        # Learned leaf features and ordered local context.
        gpu_frame = RoundedRectangle(
            width=13.65,
            height=6.18,
            corner_radius=0.14,
            stroke_color=LINE,
            stroke_width=1.2,
        ).move_to(DOWN * 0.14)
        gpu_label = geist_mono("WebGPU", 16, MUTED).move_to(gpu_frame.get_corner(UL) + RIGHT * 0.7 + DOWN * 0.4)
        next_caption = caption("Embed each part")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            FadeIn(gpu_frame),
            FadeIn(gpu_label),
            FadeOut(part_note),
            parts.animate.move_to(DOWN * 2.55),
            run_time=0.8,
        )
        current_caption = next_caption

        feature_strips = VGroup(*[
            activation_strip(cell.source, [cell.get_x(), -1.55, 0])
            for cell in parts
        ])
        channel_label = geist_mono("32 learned channels", 17, MUTED).move_to(UP * 2.42)
        self.play(
            LaggedStart(*[FadeIn(strip, shift=UP * 0.08) for strip in feature_strips], lag_ratio=0.025),
            FadeIn(channel_label),
            run_time=0.85,
        )

        local_note = geist_mono("5-part window + nearest non-space", 17, MUTED).move_to(UP * 1.88)
        scan_window = SurroundingRectangle(
            VGroup(*parts[0:5]), color=ACTIVATION, buff=0.07, stroke_width=2.4, corner_radius=0.2,
        )
        scan_end = SurroundingRectangle(
            VGroup(*parts[-5:]), color=ACTIVATION, buff=0.07, stroke_width=2.4, corner_radius=0.2,
        )
        next_caption = caption("Scan context both directions")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            FadeIn(local_note),
            Create(scan_window),
            run_time=0.55,
        )
        current_caption = next_caption
        self.play(
            Transform(scan_window, scan_end),
            run_time=1.8,
            rate_func=smooth,
        )
        scan_start = SurroundingRectangle(
            VGroup(*parts[0:5]), color=ACTIVATION, buff=0.07, stroke_width=2.4, corner_radius=0.2,
        )
        self.play(
            Transform(scan_window, scan_start),
            run_time=1.8,
            rate_func=smooth,
        )
        self.wait(0.35)

        # Pass 1 — shared pairwise tree, one level at a time.
        xs = [cell.get_x() for cell in parts]
        levels, edge_levels = make_tree(xs)
        leaf_nodes = levels[0]
        next_caption = caption("Pass 1: bottom-up")
        level_count = geist_mono("21", 17, MUTED).move_to(RIGHT * 5.82 + UP * 2.4)
        self.play(
            ReplacementTransform(current_caption, next_caption),
            ReplacementTransform(feature_strips, leaf_nodes),
            FadeOut(channel_label),
            FadeOut(local_note),
            FadeOut(scan_window),
            FadeIn(level_count),
            run_time=0.65,
        )
        current_caption = next_caption

        counts = [21, 11, 6, 3, 2, 1]
        for depth, (nodes, edges) in enumerate(zip(levels[1:], edge_levels), start=1):
            next_count = geist_mono(" → ".join(str(value) for value in counts[:depth + 1]), 17, MUTED)
            next_count.move_to(RIGHT * 5.15 + UP * 2.4)
            if next_count.get_right()[0] > 6.55:
                next_count.shift(LEFT * (next_count.get_right()[0] - 6.55))
            self.play(
                Create(edges),
                FadeIn(nodes, scale=0.78),
                ReplacementTransform(level_count, next_count),
                run_time=0.62,
            )
            level_count = next_count

        root = levels[-1][0]
        next_caption = caption("One whole-file context")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            Indicate(root, color=ACTIVATION, scale_factor=1.65),
            run_time=0.75,
        )
        current_caption = next_caption
        self.wait(0.45)

        # Pass 2 — context flows from every parent into its children.
        next_caption = caption("Pass 2: top-down")
        self.play(ReplacementTransform(current_caption, next_caption), run_time=0.4)
        current_caption = next_caption
        for depth in range(len(edge_levels) - 1, -1, -1):
            flashes = []
            for edge in edge_levels[depth]:
                parent_to_child = Line(
                    edge.get_end(),
                    edge.get_start(),
                    color=ACTIVATION,
                    stroke_width=4,
                )
                flashes.append(ShowPassingFlash(parent_to_child, time_width=0.65))
            self.play(
                AnimationGroup(*flashes, lag_ratio=0.02),
                *[node.animate.set_fill(ACTIVATION, opacity=0.22) for node in levels[depth]],
                run_time=0.62,
            )

        context_nodes = VGroup(*[
            activation_node(f"context:{cell.source}", [cell.get_x() + 0.105, -1.43, 0], width=0.14, height=0.13, cells=2)
            for cell in parts
        ])
        parents_and_edges = VGroup(*edge_levels, *levels[1:])
        next_caption = caption("Each part receives whole-file context")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            FadeOut(parents_and_edges),
            FadeOut(level_count),
            leaf_nodes.animate.shift(LEFT * 0.105),
            FadeIn(context_nodes, shift=RIGHT * 0.08),
            run_time=0.85,
        )
        current_caption = next_caption
        pair_focus = VGroup(leaf_nodes[0], context_nodes[0])
        pair_brace = Brace(pair_focus, UP, color=WHITE_TEXT, buff=0.06)
        pair_label = geist_mono("Leaf + Context", 13.5, WHITE_TEXT).next_to(pair_brace, UP, buff=0.09)
        self.play(FadeIn(pair_brace), FadeIn(pair_label), run_time=0.4)
        self.wait(0.45)

        # Show one classifier, then classify all leaves in parallel.
        network, score_rows = make_network_panel()
        selected = parts[0].copy().scale(1.22).move_to(LEFT * 6.0 + UP * network[2].get_y())
        next_caption = caption("9 scores → one predicted type")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            FadeOut(parts, shift=DOWN * 0.08),
            FadeOut(leaf_nodes),
            FadeOut(context_nodes),
            FadeOut(pair_brace),
            FadeOut(pair_label),
            FadeIn(selected),
            run_time=0.65,
        )
        current_caption = next_caption
        self.play(
            Create(network[0]),
            Create(network[1]),
            FadeIn(VGroup(*network[2:])),
            run_time=1.15,
        )
        self.play(Indicate(score_rows[4], color=SEMANTIC["keyword"], scale_factor=1.05), run_time=0.55)
        self.wait(0.35)

        parts.set_opacity(1)
        badges = make_badges(parts)
        legend = make_type_legend(["plain", "keyword", "operator", "string", "function"]).move_to(UP * 2.32)
        color_animations = []
        for cell in parts:
            color = SEMANTIC[cell.semantic]
            color_animations.extend([
                cell.glyph.animate.set_color(color),
                cell.box.animate.set_stroke(color, width=1.35),
            ])
        next_caption = caption("Classify every part in parallel")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            FadeOut(network),
            FadeOut(selected),
            FadeIn(parts, shift=UP * 0.08),
            FadeIn(badges, shift=UP * 0.08),
            FadeIn(legend),
            *color_animations,
            run_time=0.95,
        )
        current_caption = next_caption
        self.wait(0.55)

        # Adjacent labels become the final highlighted source.
        highlighted = make_highlighted_code().move_to(UP * 0.5)
        next_caption = caption("Adjacent labels → highlight spans")
        self.play(
            ReplacementTransform(current_caption, next_caption),
            ReplacementTransform(parts, highlighted),
            FadeOut(badges),
            FadeOut(legend),
            FadeOut(gpu_frame),
            FadeOut(gpu_label),
            run_time=1.0,
        )
        self.wait(2.4)
