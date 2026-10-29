#!/usr/bin/env python3
"""
isa51.py - ISA-5.1 instrument identification library (functional identification + tag numbers).

Reference: ANSI/ISA-5.1 (2009) "Instrumentation Symbols and Identification", Table 1 (identification letters).
The letter tables below were written from memory of that table. VERIFY them against your licensed copy of
the standard before relying on them for deliverables; edit the dictionaries if your project differs.

What it does
  parse_function("LIA")      -> FunctionInfo: valid?, variable, modifiers, readout/output letters, warnings
  parse_tag("13-TW-87")      -> TagInfo(area, function, loop, suffix) or None
  compose_tag("13","TW","87")-> "13-TW-87"   (style configurable)
  explain("13-LIA-45")       -> "Level - Indicate - Alarm"
  PROJECT_FUNCTIONS          -> non-ISA codes your project uses (e.g. RV = relief valve): pass as `extras`

Usage in the extractor:  from isa51 import is_valid_function, parse_function
Run `python isa51.py` to execute the self-test.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- tables (ISA-5.1-2009 Table 1)
FIRST_LETTER = {
    "A": "Analysis", "B": "Burner / combustion", "C": "User's choice", "D": "User's choice",
    "E": "Voltage", "F": "Flow rate", "G": "User's choice", "H": "Hand", "I": "Current (electrical)",
    "J": "Power", "K": "Time / schedule", "L": "Level", "M": "User's choice", "N": "User's choice",
    "O": "User's choice", "P": "Pressure / vacuum", "Q": "Quantity", "R": "Radiation",
    "S": "Speed / frequency", "T": "Temperature", "U": "Multivariable", "V": "Vibration / mechanical analysis",
    "W": "Weight / force", "X": "Unclassified", "Y": "Event / state / presence", "Z": "Position / dimension",
}
USERS_CHOICE = {k for k, v in FIRST_LETTER.items() if v == "User's choice"}   # need a legend definition

FIRST_MODIFIER = {"D": "Differential", "F": "Ratio (fraction)", "J": "Scan", "K": "Time rate of change",
                  "M": "Momentary", "Q": "Integrate / totalize", "S": "Safety",
                  "X": "X-axis", "Y": "Y-axis", "Z": "Z-axis"}

READOUT = {"A": "Alarm", "B": "User's choice", "E": "Sensor / primary element", "G": "Glass / gauge / viewing device",
           "I": "Indicate", "L": "Light", "N": "User's choice", "O": "Orifice / restriction",
           "P": "Point (test connection)", "R": "Record", "U": "Multifunction", "W": "Well / probe",
           "X": "Unclassified"}
OUTPUT = {"C": "Control", "K": "Control station", "S": "Switch", "T": "Transmit", "U": "Multifunction",
          "V": "Valve / damper / louver", "X": "Unclassified", "Y": "Relay / compute / convert",
          "Z": "Driver / actuator / final element"}
SUCC_MODIFIER = {"H": "High", "L": "Low", "M": "Middle / intermediate"}

SUCCEEDING = {**READOUT, **OUTPUT, **SUCC_MODIFIER}

# Codes that appear on real P&IDs but are not ISA 5.1 letter combinations. Extend per project.
PROJECT_FUNCTIONS: Dict[str, str] = {
    "RV": "Relief valve (project convention; ISA would read R = radiation)",
}

# Common drawing words that happen to be letter-valid. Never treat as instrument functions.
STOPWORDS = {"NOTE", "TYPE", "ONLY", "NEW", "THE", "FOR", "AND", "NOT", "FROM", "TO", "HOLD", "MESH",
             "VENT", "POT", "DRUM", "LINE", "SLOPE", "TRIP", "RESET", "FLOW", "REV", "DWG", "NO", "SIZE",
             "SHT", "AREA", "LOOP", "TAG"}

# ISA-5.1 instrument bubble conventions (location is shown by the line through the symbol)
BUBBLE_SHAPES = {
    "circle": "Discrete instrument",
    "circle_in_square": "Shared display / shared control (DCS)",
    "hexagon": "Computer function",
    "diamond_in_square": "Programmable logic controller (PLC)",
    "square_diamond": "Interlock logic",
}
BUBBLE_LOCATION = {
    "none": "Field mounted",
    "solid_single": "Primary location, normally accessible to operator (e.g. central control room)",
    "solid_double": "Auxiliary location, normally accessible to operator (e.g. local panel)",
    "dashed_single": "Primary location, not normally accessible to operator (behind panel)",
    "dashed_double": "Auxiliary location, not normally accessible to operator",
}


# --------------------------------------------------------------------------- function letters
@dataclass
class FunctionInfo:
    code: str
    valid: bool
    source: str = ""                 # "isa" | "project" | ""
    variable: str = ""
    modifiers: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def description(self):
        parts = [self.variable] + self.modifiers + self.functions
        return " - ".join(p for p in parts if p)


def parse_function(code: str, extras: Optional[Dict[str, str]] = None) -> FunctionInfo:
    """Interpret a function-letter string such as TW, LIA, PDIT, PSV, TSHH. Never raises."""
    code = (code or "").strip().upper()
    extras = extras if extras is not None else PROJECT_FUNCTIONS
    if code in extras:
        return FunctionInfo(code, True, "project", variable=extras[code])
    if not re.fullmatch(r"[A-Z]{1,5}", code) or code in STOPWORDS:
        return FunctionInfo(code, False, warnings=["not 1-5 letters, or a common drawing word"])

    info = FunctionInfo(code, True, "isa", variable=FIRST_LETTER[code[0]])
    if code[0] in USERS_CHOICE:
        info.warnings.append(f"first letter {code[0]} is 'user's choice': needs a project legend definition")
    rest = code[1:]

    # first-letter modifier (D, F, J, K, Q always; S only when followed by V/E/Z, as in PSV, PSE)
    if rest:
        m = rest[0]
        takes_mod = m in "DFJKQ" or (m == "S" and len(rest) > 1 and rest[1] in "VEZ")
        if takes_mod:
            info.modifiers.append(FIRST_MODIFIER[m])
            rest = rest[1:]

    seen_modifier = False
    for ch in rest:
        if ch not in SUCCEEDING:
            info.valid = False
            info.warnings.append(f"letter {ch} is not a valid succeeding letter")
            continue
        if ch in SUCC_MODIFIER:
            seen_modifier = True
            info.modifiers.append(SUCC_MODIFIER[ch])
        else:
            if seen_modifier:
                info.warnings.append("function letter after a High/Low/Middle modifier (unusual order)")
            kind = READOUT.get(ch) if ch in READOUT and ch not in OUTPUT else OUTPUT.get(ch, READOUT.get(ch))
            info.functions.append(kind)
    if not rest and not info.modifiers:
        info.warnings.append("variable only, no function letter (valid for e.g. H, but check the symbol)")
    return info


def is_valid_function(code: str, extras: Optional[Dict[str, str]] = None) -> bool:
    return parse_function(code, extras).valid


# --------------------------------------------------------------------------- tag numbers
@dataclass
class TagInfo:
    tag: str
    area: str
    function: str
    loop: str
    suffix: str
    info: FunctionInfo


_TAG_RX = re.compile(r'^(?:(?P<area>[0-9A-Z]{1,4})-)?(?P<func>[A-Z]{1,5})[- ](?P<loop>\d{1,5})(?P<suffix>[A-Z]?)$')

def parse_tag(text: str, extras: Optional[Dict[str, str]] = None, min_letters: int = 2) -> Optional[TagInfo]:
    """Parse '13-TW-87', 'PT-1001A', 'FIC 101'. Returns None unless the function letters are valid.
    min_letters=2 keeps one-letter equipment tags (P-1001, E-101) from being read as instruments."""
    t = (text or "").strip().upper().strip(",;.")
    m = _TAG_RX.match(t)
    if not m:
        return None
    func = m.group("func")
    if len(func) < min_letters:
        return None
    info = parse_function(func, extras)
    if not info.valid:
        return None
    return TagInfo(t, m.group("area") or "", func, m.group("loop"), m.group("suffix"), info)


def compose_tag(area: str, function: str, loop: str, suffix: str = "", style: str = "area-func-loop") -> str:
    """style: 'area-func-loop' -> 13-TW-87 | 'func-loop' -> TW-87 | 'func loop' -> TW 87"""
    if style == "area-func-loop" and area:
        return f"{area}-{function}-{loop}{suffix}"
    if style == "func loop":
        return f"{function} {loop}{suffix}"
    return f"{function}-{loop}{suffix}"


def explain(tag: str, extras: Optional[Dict[str, str]] = None) -> str:
    t = parse_tag(tag, extras, min_letters=1)
    return t.info.description if t else "not a valid ISA-5.1 / project instrument tag"


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    ok = {"TW": True, "LIA": True, "PDIT": True, "PSV": True, "TSHH": True, "FIC": True, "HV": True, "HCV": True,
          "XX": True, "PG": True, "FQI": True, "RV": True, "LG": True, "HY": True, "H": True}
    bad = {"NOTE": False, "TO": True}      # NOTE = stopword; 'TO' is letter-valid (T + O) -> caught by layout/stoplist
    for c, exp in {**ok, "NOTE": False}.items():
        assert is_valid_function(c) == exp, (c, parse_function(c))
    assert not is_valid_function("PPPPPP") and not is_valid_function("P1")
    assert parse_function("PSV").modifiers == ["Safety"]
    assert parse_function("PSH").functions == ["Switch"] and parse_function("PSH").modifiers == ["High"]
    t = parse_tag("13-TW-87"); assert (t.area, t.function, t.loop) == ("13", "TW", "87")
    t = parse_tag("PT-1001A"); assert (t.area, t.function, t.loop, t.suffix) == ("", "PT", "1001", "A")
    assert parse_tag("P-1001") is None and parse_tag("13-QQQQQQ-1") is None
    assert compose_tag("13", "TW", "87") == "13-TW-87"
    print("self-test passed")
    for tag in ("13-LIA-45", "13-TW-87", "13-RV-63", "PDIT-210", "13-PSV-77"):
        print(f"{tag:12s} {explain(tag)}")
