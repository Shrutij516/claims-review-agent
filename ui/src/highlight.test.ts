import { describe, expect, it } from "vitest";
import { collapse, highlightQuote, highlightQuotes } from "./highlight";

describe("highlightQuote", () => {
  it("finds a quote across whitespace differences and marks only the matched part", () => {
    const note = "Patient has knee pain.\n  No physical   therapy has been tried. Exam is mild.";
    const { segments, notFound } = highlightQuote(note, "No physical therapy  has\nbeen tried.");
    expect(notFound).toEqual([]);
    expect(segments.map((s) => s.highlight)).toEqual([false, true, false]);
    expect(segments[1].text).toBe("No physical   therapy has been tried.");
    expect(segments.map((s) => s.text).join("")).toBe(note);
  });

  it("returns the whole text unmarked and reports a quote that is not found", () => {
    const note = "Exam shows mild tenderness.";
    const { segments, notFound } = highlightQuote(note, "severe swelling");
    expect(segments).toEqual([{ text: note, highlight: false }]);
    expect(notFound).toEqual(["severe swelling"]);
  });

  it("is case sensitive like the server check and treats an empty quote as not found", () => {
    expect(highlightQuote("Knee pain", "knee pain").notFound).toEqual(["knee pain"]);
    expect(highlightQuote("Knee pain", "   ").notFound).toEqual(["   "]);
  });

  it("marks several quotes and merges overlapping ones", () => {
    const { segments } = highlightQuotes("a b c d e", ["b c", "c d", "e"]);
    expect(segments).toEqual([
      { text: "a ", highlight: false },
      { text: "b c d", highlight: true },
      { text: " ", highlight: false },
      { text: "e", highlight: true },
    ]);
  });

  it("collapse trims and squeezes whitespace", () => {
    expect(collapse("  a \n\t b  ")).toBe("a b");
  });
});
