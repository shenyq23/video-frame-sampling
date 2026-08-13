import type { NumericSpec } from "../numeric";
import { numericPlaceholder } from "../numeric";

interface Props {
  spec: NumericSpec;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Text-based numeric input: no spinner arrows, and clearing the box leaves it
 * empty (showing the grey example) instead of snapping back to 0.
 */
export function NumberField({ spec, value, onChange, disabled, className }: Props) {
  return (
    <label className={className ?? "field"}>
      <span>{spec.label}</span>
      <input
        type="text"
        inputMode={spec.kind === "integer" ? "numeric" : "decimal"}
        autoComplete="off"
        disabled={disabled}
        value={value}
        placeholder={numericPlaceholder(spec)}
        onChange={(event) => onChange(event.target.value)}
      />
      {spec.hint && <small>{spec.hint}</small>}
    </label>
  );
}
