#!/usr/bin/env python3
"""Render the deterministic current-main Figure 2(b) reproduction as SVG."""

import html
import json
import sys
from pathlib import Path


PRE_SURGERY_RECORD = 47.824


def fail(message: str) -> None:
    raise SystemExit(f"plot_figure2b.py: FAIL: {message}")


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: plot_figure2b.py ARTIFACT.json OUTPUT.svg")
    source, destination = map(Path, sys.argv[1:])
    artifact = json.loads(source.read_text())
    rows = {row.get("arm"): row for row in artifact.get("results", [])}
    if artifact.get("format") != 2 or artifact.get("set") != "figure2b":
        fail("expected format 2, set figure2b")
    if set(rows) != {"treesa", "treesa_rounds"}:
        fail("expected treesa and treesa_rounds rows")
    curve = rows["treesa_rounds"].get("curve", [])
    if not curve:
        fail("treesa_rounds row has no curve")

    width, height = 900, 560
    left, right, top, bottom = 92, 28, 76, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_values = [rows["treesa"]["tc"], PRE_SURGERY_RECORD]
    for point in curve:
        y_values.extend(
            [
                point["tc_before"],
                point["tc_after_surgery"],
                point["tc_after_anneal"],
                point["tc_retained"],
            ]
        )
    y_min = min(y_values) - 0.25
    y_max = max(y_values) + 0.35
    rounds = len(curve)

    def x(round_index: float) -> float:
        return left + round_index / rounds * plot_width

    def y(tc: float) -> float:
        return top + (y_max - tc) / (y_max - y_min) * plot_height

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Figure 2(b) reproduction from current main</title>",
        "<desc>Surface-code distance 21 contraction cost over 32 deterministic surgery and cold fine-tuning rounds. Raw fine-tuning endpoints are separated from the retained incumbent.</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Helvetica,Arial,sans-serif" fill="#111">',
        '<text x="92" y="29" font-size="20" font-weight="700">Figure 2(b) reproduction — current main</text>',
        '<text x="92" y="52" font-size="13" fill="#444">surfacecode_d21; deterministic 32-round work budget; TreeSA defaults</text>',
    ]

    for tick in range(48, 58, 2):
        yy = y(tick)
        svg.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e4e4e4" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" font-size="12">{tick}</text>'
        )
    for tick in range(0, rounds + 1, 4):
        xx = x(tick)
        svg.append(
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{xx:.2f}" y="{height-bottom+23}" text-anchor="middle" font-size="12">{tick}</text>'
        )

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111" stroke-width="1.4"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111" stroke-width="1.4"/>',
            f'<text x="{left + plot_width/2:.2f}" y="{height-24}" text-anchor="middle" font-size="14">completed surgery/fine-tuning rounds</text>',
            f'<text x="23" y="{top + plot_height/2:.2f}" text-anchor="middle" font-size="14" transform="rotate(-90 23 {top + plot_height/2:.2f})">contraction cost tc  [log₂ flops]</text>',
        ]
    )

    record_y = y(PRE_SURGERY_RECORD)
    svg.append(
        f'<line x1="{left}" y1="{record_y:.2f}" x2="{width-right}" y2="{record_y:.2f}" stroke="#222" stroke-width="1.5" stroke-dasharray="3 5"/>'
    )

    retained = [(0.0, rows["treesa"]["tc"])]
    retained.extend((point["round"] + 1.0, point["tc_retained"]) for point in curve)
    points = " ".join(f"{x(xx):.2f},{y(yy):.2f}" for xx, yy in retained)
    svg.append(
        f'<polyline points="{points}" fill="none" stroke="#0072b2" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    for point in curve:
        xx = x(point["round"] + 1.0)
        endpoint_y = y(point["tc_after_anneal"])
        svg.append(
            f'<circle cx="{xx:.2f}" cy="{endpoint_y:.2f}" r="3.1" fill="white" stroke="#777" stroke-width="1.3"/>'
        )
        if point["surgery_accepted"]:
            surgery_y = y(point["tc_after_surgery"])
            svg.extend(
                [
                    f'<line x1="{xx-5:.2f}" y1="{surgery_y-5:.2f}" x2="{xx+5:.2f}" y2="{surgery_y+5:.2f}" stroke="#cc3311" stroke-width="2.2"/>',
                    f'<line x1="{xx-5:.2f}" y1="{surgery_y+5:.2f}" x2="{xx+5:.2f}" y2="{surgery_y-5:.2f}" stroke="#cc3311" stroke-width="2.2"/>',
                ]
            )

    legend_x, legend_y = 520, 91
    svg.extend(
        [
            f'<rect x="{legend_x-12}" y="{legend_y-18}" width="360" height="105" fill="white" fill-opacity="0.92" stroke="#bbb"/>',
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" stroke="#0072b2" stroke-width="2.8"/><text x="{legend_x+38}" y="{legend_y+4}" font-size="12">retained incumbent</text>',
            f'<circle cx="{legend_x+14}" cy="{legend_y+22}" r="3.1" fill="white" stroke="#777" stroke-width="1.3"/><text x="{legend_x+38}" y="{legend_y+26}" font-size="12">raw fine-tuning endpoint</text>',
            f'<path d="M {legend_x+9} {legend_y+39} l 10 10 m -10 0 l 10 -10" stroke="#cc3311" stroke-width="2.2"/><text x="{legend_x+38}" y="{legend_y+48}" font-size="12">accepted waist rebuild</text>',
            f'<line x1="{legend_x}" y1="{legend_y+66}" x2="{legend_x+28}" y2="{legend_y+66}" stroke="#222" stroke-width="1.5" stroke-dasharray="3 5"/><text x="{legend_x+38}" y="{legend_y+70}" font-size="12">pre-surgery record 47.824</text>',
            f'<text x="{left}" y="{height-5}" font-size="10" fill="#555">Source: {esc(source.name)}; final retained tc = {rows["treesa_rounds"]["tc"]:.6f}</text>',
            "</g></svg>",
        ]
    )
    destination.write_text("\n".join(svg) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
