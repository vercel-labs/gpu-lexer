type SyntaxClassName =
  | "plain"
  | "comment"
  | "string"
  | "number"
  | "keyword"
  | "type"
  | "function"
  | "constant"
  | "operator";

interface SyntaxSpan {
  type: SyntaxClassName;
  start: number;
  end: number;
  /**
   * How sure the model is of `type`, in sixteenths from 1/16 to 1. It is the
   * lowest value among the parts merged into this span, and a ranking score
   * rather than a calibrated probability.
   */
  confidence: number;
}

export declare function parse(
  code: string,
): Promise<SyntaxSpan[]>;
