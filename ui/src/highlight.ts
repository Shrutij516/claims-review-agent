// Find model quotes inside the clinical note, the same way the server checks them:
// exact match after collapsing every run of whitespace to one space and trimming the ends.

export interface Segment {
  text: string;
  highlight: boolean;
}

export interface HighlightResult {
  segments: Segment[];
  notFound: string[];
}

export function collapse(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(" ");
}

/** The collapsed text plus, for each collapsed character, its index in the original text. */
function collapseWithMap(text: string): { collapsed: string; map: number[] } {
  let collapsed = "";
  const map: number[] = [];
  let pendingSpace = -1;
  for (let i = 0; i < text.length; i++) {
    if (/\s/.test(text[i])) {
      if (collapsed.length > 0 && pendingSpace < 0) pendingSpace = i;
      continue;
    }
    if (pendingSpace >= 0) {
      collapsed += " ";
      map.push(pendingSpace);
      pendingSpace = -1;
    }
    collapsed += text[i];
    map.push(i);
  }
  return { collapsed, map };
}

/** Split text into segments with every found quote marked. Quotes that are not found are returned in notFound. */
export function highlightQuotes(text: string, quotes: string[]): HighlightResult {
  const { collapsed, map } = collapseWithMap(text);
  const ranges: Array<[number, number]> = [];
  const notFound: string[] = [];
  for (const quote of quotes) {
    const q = collapse(quote);
    const at = q ? collapsed.indexOf(q) : -1;
    if (at < 0) {
      notFound.push(quote);
      continue;
    }
    ranges.push([map[at], map[at + q.length - 1] + 1]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([r[0], r[1]]);
  }
  const segments: Segment[] = [];
  let pos = 0;
  for (const [start, end] of merged) {
    if (start > pos) segments.push({ text: text.slice(pos, start), highlight: false });
    segments.push({ text: text.slice(start, end), highlight: true });
    pos = end;
  }
  if (pos < text.length || segments.length === 0) segments.push({ text: text.slice(pos), highlight: false });
  return { segments, notFound };
}

/** Convenience wrapper for one quote. */
export function highlightQuote(text: string, quote: string): HighlightResult {
  return highlightQuotes(text, [quote]);
}
