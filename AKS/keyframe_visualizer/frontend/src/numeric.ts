export type NumericKind = "integer" | "decimal";

export interface NumericSpec {
  /** Field label, reused verbatim in validation messages. */
  label: string;
  kind: NumericKind;
  /** Value used when the box is left empty; also the pre-filled value. */
  default: number;
  min?: number;
  max?: number;
  /** ``true`` turns the corresponding bound into a strict inequality. */
  exclusiveMin?: boolean;
  exclusiveMax?: boolean;
  hint?: string;
}

export type NumericSpecMap = Record<string, NumericSpec>;

// Native number inputs coerce a cleared box to 0, so "delete then retype" ends
// up as "01.2".  Every numeric field is a plain text input instead; these
// patterns replace the browser's own parsing.
const INTEGER_PATTERN = /^[+-]?\d+$/;
const DECIMAL_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/;

export function describeNumericSpec(spec: NumericSpec): string {
  const noun = spec.kind === "integer" ? "整数" : "数字";
  const bounds: string[] = [];
  if (spec.min !== undefined) {
    bounds.push(spec.exclusiveMin ? `大于 ${spec.min}` : `不小于 ${spec.min}`);
  }
  if (spec.max !== undefined) {
    bounds.push(spec.exclusiveMax ? `小于 ${spec.max}` : `不大于 ${spec.max}`);
  }
  return bounds.length ? `${bounds.join("、")} 的${noun}` : noun;
}

export function numericPlaceholder(spec: NumericSpec): string {
  return `例如：${spec.default}`;
}

export type NumericResult =
  | { ok: true; value: number }
  | { ok: false; message: string };

export function resolveNumeric(spec: NumericSpec, raw: string): NumericResult {
  const text = raw.trim();
  if (!text) return { ok: true, value: spec.default };
  const pattern = spec.kind === "integer" ? INTEGER_PATTERN : DECIMAL_PATTERN;
  const value = Number(text);
  if (!pattern.test(text) || !Number.isFinite(value)) {
    return { ok: false, message: `${spec.label}必须是${describeNumericSpec(spec)}。` };
  }
  if (spec.min !== undefined && (spec.exclusiveMin ? value <= spec.min : value < spec.min)) {
    return { ok: false, message: `${spec.label}必须是${describeNumericSpec(spec)}。` };
  }
  if (spec.max !== undefined && (spec.exclusiveMax ? value >= spec.max : value > spec.max)) {
    return { ok: false, message: `${spec.label}必须是${describeNumericSpec(spec)}。` };
  }
  return { ok: true, value };
}

export type NumericInputs = Record<string, string>;

export function numericDefaults<M extends NumericSpecMap>(specs: M): Record<keyof M, number> {
  const values = {} as Record<keyof M, number>;
  for (const key of Object.keys(specs) as Array<keyof M>) {
    values[key] = specs[key].default;
  }
  return values;
}

export function numericInputDefaults<M extends NumericSpecMap>(specs: M): NumericInputs {
  const inputs: NumericInputs = {};
  for (const [key, spec] of Object.entries(specs)) {
    inputs[key] = String(spec.default);
  }
  return inputs;
}

/** Pre-fill the boxes from a session snapshot, falling back to each default. */
export function numericInputsFrom<M extends NumericSpecMap>(
  specs: M,
  snapshot: Record<string, unknown> | undefined | null,
): NumericInputs {
  const inputs = numericInputDefaults(specs);
  if (!snapshot) return inputs;
  for (const key of Object.keys(specs)) {
    const value = snapshot[key];
    if (typeof value === "number" && Number.isFinite(value)) inputs[key] = String(value);
  }
  return inputs;
}

export type NumericFieldsResult<M extends NumericSpecMap> =
  | { ok: true; values: Partial<Record<keyof M, number>> }
  | { ok: false; message: string };

/**
 * Validate every spec'd box, reporting the first problem in declaration order.
 * Keys in ``skip`` belong to disabled inputs and keep their current value.
 */
export function resolveNumericFields<M extends NumericSpecMap>(
  specs: M,
  inputs: NumericInputs,
  skip: Iterable<keyof M> = [],
): NumericFieldsResult<M> {
  const skipped = new Set<keyof M>(skip);
  const values: Partial<Record<keyof M, number>> = {};
  for (const key of Object.keys(specs) as Array<keyof M>) {
    if (skipped.has(key)) continue;
    const result = resolveNumeric(specs[key], inputs[key as string] ?? "");
    if (!result.ok) return { ok: false, message: result.message };
    values[key] = result.value;
  }
  return { ok: true, values };
}
