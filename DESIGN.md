---
name: regressio
description: A browser proof suite drawn as a pen-plotter acceptance sheet.
colors:
  plotting-blue: "#1239c5"
  plotting-blue-light: "#94a9eb"
  system-green: "#16702a"
  safety-orange: "#d63c12"
  drafting-paper: "#f8f8f8"
  raised-paper: "#ffffff"
  graphite: "#171916"
  graphite-muted: "#5f645f"
  rule: "#c4c9c2"
  rule-strong: "#8b918a"
typography:
  display:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "clamp(2.5rem, 6vw, 5.5rem)"
    fontWeight: 700
    lineHeight: 0.94
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "1.08rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.05em"
  body:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "normal"
  label:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.08em"
  readout:
    fontFamily: "SFMono-Regular, Consolas, Liberation Mono, monospace"
    fontSize: "clamp(0.7rem, 0.72vw, 0.82rem)"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "normal"
  wordmark:
    fontFamily: "Delius Unicase, sans-serif"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.05em"
rounded:
  square: "0"
  round: "50%"
spacing:
  xs: "0.35rem"
  sm: "0.65rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "3rem"
components:
  button-primary:
    backgroundColor: "{colors.graphite}"
    textColor: "{colors.raised-paper}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "0.65rem 0.85rem"
  button-primary-hover:
    backgroundColor: "{colors.plotting-blue}"
    textColor: "{colors.raised-paper}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "0.65rem 0.85rem"
  button-copy-success:
    backgroundColor: "{colors.system-green}"
    textColor: "{colors.raised-paper}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "0.45rem 0.7rem"
  button-copy-error:
    backgroundColor: "{colors.safety-orange}"
    textColor: "{colors.raised-paper}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "0.45rem 0.7rem"
  tab-family:
    backgroundColor: "transparent"
    textColor: "{colors.graphite}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    height: "3.5rem"
  tab-family-selected:
    backgroundColor: "{colors.plotting-blue}"
    textColor: "{colors.raised-paper}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    height: "3.5rem"
  proof-readout:
    backgroundColor: "{colors.drafting-paper}"
    textColor: "{colors.graphite}"
    typography: "{typography.readout}"
    rounded: "{rounded.square}"
    padding: "0 1rem"
---

# Design System: regressio

## Overview

**Creative North Star: "The Plotter Acceptance Sheet"**

regressio feels like a calibrated technical sheet caught midway through a proof run: cool paper, exact graphite rulings, measured readouts, and a saturated blue pen trace. Information stays exposed and continuous. The visual hierarchy comes from scale, alignment, and ruled partitions instead of a collection of floating dashboard cards.

The system is dense but calm. Large condensed headlines establish the claim, monospaced measurements make evidence scannable, and the realistic plotter carriage gives the otherwise flat world one physical signature. Every expressive move should help a developer see the suite pass, follow the trace, or inspect the underlying results.

**Key Characteristics:**

- Cool drafting-paper ground with graphite structure.
- Saturated plotting blue for active traces, selections, and positive proof emphasis.
- Safety orange reserved for failures and the keyboard focus ring.
- Condensed sans-serif hierarchy paired with tabular monospaced evidence.
- Registration marks, calibration ticks, ruled tabs, and plots without card shells.
- A single plotter carriage supplies the system's realistic material accent.
- State-driven motion separates calibration, proof drawing, and family inspection.

## Colors

The palette reads like drafting media: near-white paper, low-chroma graphite and rules, one decisive blue ink, and one exceptional warning color.

### Primary

- **Plotting Blue** (`#1239c5`): draws the main regression trace and marks active or selected controls.
- **Light Plotting Blue** (`#94a9eb`): distinguishes secondary traces without competing with the primary line.

### Secondary

- **System Green** (`#16702a`): confirms a healthy runtime and a successful clipboard write.

### Tertiary

- **Safety Orange** (`#d63c12`): signals proof and clipboard failures and forms the highly visible keyboard focus outline.

### Neutral

- **Drafting Paper** (`#f8f8f8`): continuous page ground for the proof sheet and labs.
- **Raised Paper** (`#ffffff`): point interiors and light-on-dark text; it is a tonal aid rather than a floating surface.
- **Graphite** (`#171916`): primary text, strong dividers, dark controls, and the footer field.
- **Muted Graphite** (`#5f645f`): supporting copy, measurement labels, table content, and residual notes.
- **Rule** (`#c4c9c2`): chart grids, table rows, and quiet partitions.
- **Strong Rule** (`#8b918a`): structural divisions, tab edges, and calibration marks.

### Named Rules

**The One Ink Rule.** Plotting blue carries the active trace and current selection; keep it scarce enough that the live result remains the clearest blue object.

**The Safety Exception Rule.** Safety orange appears only when the interface requires immediate attention: a failed proof or a visible keyboard focus target.

**The Continuous Sheet Rule.** Separate regions with graphite rules and alignment before introducing another surface color.

## Typography

**Display Font:** Barlow Condensed (with Arial Narrow and sans-serif fallbacks)
**Body Font:** Barlow Condensed (with Arial Narrow and sans-serif fallbacks)
**Label/Mono Font:** SFMono-Regular (with Consolas and Liberation Mono fallbacks)
**Wordmark Font:** Delius Unicase (with sans-serif fallback)

**Character:** The condensed sans face gives headings and controls the economy of equipment labels. Monospaced readouts make values and code feel measured, while the hand-drawn wordmark gives the otherwise precise system a small human signature.

### Hierarchy

- **Display** (700, `clamp(2.5rem, 6vw, 5.5rem)`, 0.94): section claims, kept short and balanced within roughly 12 characters per line.
- **Title** (700, `1.08rem`, 1): plot and dataset headings that need more voice than labels without becoming page headlines.
- **Body** (400, `1rem`, 1.45): explanations and footer copy; supporting introductions stay around 52 characters wide.
- **Label** (800, `0.75rem`, `0.08em`, uppercase): buttons, family tabs, table captions, and compact section markers.
- **Readout** (400, `clamp(0.7rem, 0.72vw, 0.82rem)`, tabular figures): metrics, timestamps, equations, table values, and plot annotations.
- **Wordmark** (700, `30px`, `0.05em`): the regressio name only.

### Named Rules

**The Measured Evidence Rule.** Numbers, equations, timings, and runtime states use the monospaced readout face with tabular figures.

**The Condensed Claim Rule.** Large statements use a tight condensed display; do not expand them into wide, soft marketing typography.

## Layout

The wide proof sheet uses a measured 80/20 composition: the unboxed plot occupies the dominant field and the proof rail aligns vertically at the right. A narrow instrument rail spans the top. The labs continue the same sheet logic with one six-part family selector above a roughly two-thirds results table and one-third code proof.

Spacing follows a compact instrument rhythm for labels and controls, then opens sharply around the large lab statement. The recurring scale runs from an extra-small inline gap through compact control padding, one-rem partitions, one-and-a-half-rem section padding, and three-rem major gaps.

At `900px`, absolute proof-sheet regions return to document flow: the header becomes two columns, the plot becomes a 4:3 block, the proof rail stacks below it, family controls use three columns, and the lab becomes one column. At `560px`, the header becomes one column, the plot becomes square, family controls use two columns, and the footer stacks.

**The Dominant Plot Rule.** On wide screens, reserve roughly four-fifths of the proof sheet for the drawing field and one-fifth for vertically aligned evidence.

**The Stack Without Cards Rule.** Responsive layouts preserve ruled continuity by stacking the same regions and borders; they do not turn those regions into detached cards.

## Elevation & Depth

The system is flat by default. Paper regions, tabs, proof blocks, and tables use tonal contrast and one-pixel rules rather than ambient shadows. The plotter carriage alone receives a soft contact shadow so it reads as a physical mechanism resting above the drawn chart.

### Shadow Vocabulary

- **Plotter Contact** (`drop-shadow(0 8px 7px rgba(23, 25, 22, 0.18))`): reserved for the transparent carriage asset over the primary plot.

### Named Rules

**The One Physical Object Rule.** Realistic depth belongs to the plotter carriage; all interface surfaces remain flat and ruled.

## Shapes

Controls, code fields, proof blocks, and tabs use square corners. Fine one-pixel borders and crosshair registration marks provide the recurring geometry. Circular form is limited to machine-like details such as the calibration knob, system lamp, and running indicator.

**The Instrument Geometry Rule.** Use straight edges for information and circles for instrument state or calibration details.

## Components

### Buttons

- **Shape:** square with a one-pixel graphite border and compact horizontal padding.
- **Primary:** graphite field, raised-paper label, heavy uppercase condensed type, and a small mechanical marker.
- **Hover / Focus:** hover shifts the field and border to plotting blue; keyboard focus uses a three-pixel safety-orange outline with a four-pixel offset.
- **Disabled / Running:** lower opacity communicates the pending state, while the triangular marker becomes a rotating outlined ring; reduced-motion mode removes the spin.

### Tabs

- **Style:** family choices form one continuous ruled row, with square cells, uppercase condensed labels, and a minimum height of `3.5rem`.
- **State:** the selected tab becomes solid plotting blue with raised-paper text, while hover changes the label to blue before selection.
- **Activation:** every selected tab replays a `480ms` control punch, compressing briefly to 96% scale between a raised-paper inset and blue outer rule.
- **Panel Transition:** changing families replays a `520ms` fast-out entrance from `0.6rem` below and 25% opacity. A blue scan crosses the panel over `680ms`; rows print from `1.25rem` left over `360ms`, delayed by `120ms` plus `36ms` per row; the code field reveals from left to right over `560ms`.
- **Responsive:** six columns become three at the medium breakpoint and two on small screens.

### Tables / Containers

- **Corner Style:** square throughout.
- **Background:** continuous drafting paper; the code proof alone uses a deep ink field.
- **Shadow Strategy:** none.
- **Border:** one-pixel rules divide rows and major regions.
- **Internal Padding:** compact table cells use roughly one-third to three-quarters of a rem; major lab regions use up to three rem.

### Proof Readout

Each metric pairs a small uppercase monospaced label with a larger monospaced value. Readouts share a vertical rail, align on rules, and use tabular figures. Fallback and failure values switch to safety orange; ordinary pass evidence remains graphite or plotting blue according to emphasis.

### Plot and Carriage

Plots remain unboxed on the drafting ground. Fine rule grids, hollow graphite points, distinct solid/dashed traces, and a compact equation key explain the drawing directly.

Starting a run changes the root state to `running` for at least `900ms`. During that calibration pass, a one-pixel blue sweep crosses the graph over `900ms` linearly, the robust dashed trace advances every `220ms`, and the carriage oscillates by two pixels over `160ms` with an alternating ease-in-out cycle. A blue “plotting / live” flag enters over `120ms`, and the run-button marker spins every `750ms`.

Completion changes the root state to `passed`. The model traces draw over `1.25s`, the carriage settles over `700ms`, and each observation reprints from 30% scale and zero opacity over `280ms`. Point delays begin at `180ms` and advance by `16ms` per point, producing a measured left-to-right sequence. A failed state does not play the completion sequence.

Under reduced motion, sweep, dashed calibration, point and row printing, panel entry and scan, code reveal, control punch, carriage calibration and settling, button spin, and copy feedback are disabled. Trace offsets and point opacity resolve immediately, and the live flag changes state without transition.

**The State-Machine Motion Rule.** Motion must explain a real state change: calibration belongs to `running`, drawing and printing belong to `passed`, family feedback belongs to a changed selection, and copy feedback belongs to clipboard state.

### Code Proof

Executed code sits in a square, deep blue-black field with pale blue text and generous internal padding. It remains visually attached to the ruled lab panel and pairs with a compact copy action. Copying compresses the button over `260ms`; success changes the label to “Copied,” turns the field system green, and confirms over `620ms`; failure changes the label to “Select code,” turns the field safety orange, and shakes it over `320ms`. Each result resets to “Copy code” after `1.2s`.

## Do's and Don'ts

### Do:

- **Do** keep evidence visible: pair claims with live values, equations, timings, or status.
- **Do** use one-pixel graphite rules to create hierarchy across continuous paper.
- **Do** reserve plotting blue for the active trace, selected controls, and explicit proof emphasis.
- **Do** use monospaced tabular figures for measurements and results.
- **Do** keep running, passed, failed, family-selection, and clipboard motion tied to their semantic states, with an immediate reduced-motion path.
- **Do** let the plotter carriage be the one realistic, shadowed signature.

### Don't:

- **Don't** break proof surfaces into rounded, floating dashboard cards.
- **Don't** add decorative shadows to tabs, tables, buttons, or proof blocks.
- **Don't** use safety orange as a general accent or decorative highlight.
- **Don't** crowd the sheet with multiple realistic machines or competing hero objects.
- **Don't** replace ruled structure with soft containers, gradients, or ornamental chrome.
