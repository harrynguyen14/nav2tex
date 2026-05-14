import re


_OVER_PATTERN = re.compile(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*?)\\over\s+([^{}]*(?:\{[^{}]*\}[^{}]*)*?)\}')

def _normalize_over(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _OVER_PATTERN.sub(
            lambda m: r'\frac{' + m.group(1).strip() + '}{' + m.group(2).strip() + '}', s
        )
    return s


def _normalize_frac_variants(s: str) -> str:
    s = re.sub(r'\\[tdc]frac\b', r'\\frac', s)
    return s


def _strip_delimiters(s: str) -> str:
    s = s.strip()
    for open_, close in (
        (r'\[', r'\]'),
        (r'\(', r'\)'),
    ):
        pat = r'^\\\[' if open_ == r'\[' else r'^\\\('
        end = r'\\\]$' if close == r'\]' else r'\\\)$'
        s = re.sub(pat + r'\s*(.*?)\s*' + end, r'\1', s, flags=re.DOTALL)
    if s.startswith('$$') and s.endswith('$$'):
        s = s[2:-2].strip()
    elif s.startswith('$') and s.endswith('$') and len(s) > 2:
        s = s[1:-1].strip()
    return s


def _normalize_font_commands(s: str) -> str:
    s = re.sub(r'\{\\bf\s+([^}]*)\}',    r'\\mathbf{\1}',   s)
    s = re.sub(r'\{\\it\s+([^}]*)\}',    r'\\mathit{\1}',   s)
    s = re.sub(r'\{\\rm\s+([^}]*)\}',    r'\\mathrm{\1}',   s)
    s = re.sub(r'\{\\cal\s+([^}]*)\}',   r'\\mathcal{\1}',  s)
    s = re.sub(r'\{\\tt\s+([^}]*)\}',    r'\\mathtt{\1}',   s)
    s = re.sub(r'\{\\sf\s+([^}]*)\}',    r'\\mathsf{\1}',   s)
    s = re.sub(r'\{\\bm\s+([^}]*)\}',    r'\\mathbf{\1}',   s)
    s = re.sub(r'\\boldsymbol\{([^}]*)\}', r'\\mathbf{\1}', s)
    s = re.sub(r'\\bm\{([^}]*)\}',        r'\\mathbf{\1}',  s)
    return s


_SPACE_PAT = re.compile(r'\\[;:!]|\\[ ]|\\thinspace|\\medspace|\\thickspace|\\negthinspace|\\negmedspace|\\negthickspace')

def _normalize_spacing(s: str) -> str:
    s = _SPACE_PAT.sub(r'\\,', s)
    s = re.sub(r'\\qquad\b', r'\\quad', s)
    return s


def _normalize_left_right(s: str) -> str:
    s = re.sub(r'\\left\s*\.', '', s)
    s = re.sub(r'\\right\s*\.', '', s)
    return s


_UNICODE_MAP = [
    ('\u2212', '-'),
    ('\u00d7', '\\times'),
    ('\u00f7', '\\div'),
    ('\u2264', '\\leq'),
    ('\u2265', '\\geq'),
    ('\u2260', '\\neq'),
    ('\u221e', '\\infty'),
    ('\u2248', '\\approx'),
    ('\u2261', '\\equiv'),
    ('\u2282', '\\subset'),
    ('\u2283', '\\supset'),
    ('\u2286', '\\subseteq'),
    ('\u2287', '\\supseteq'),
    ('\u2208', '\\in'),
    ('\u2209', '\\notin'),
    ('\u2227', '\\wedge'),
    ('\u2228', '\\vee'),
    ('\u2200', '\\forall'),
    ('\u2203', '\\exists'),
    ('\u2207', '\\nabla'),
    ('\u2202', '\\partial'),
    ('\u222b', '\\int'),
    ('\u220f', '\\prod'),
    ('\u2211', '\\sum'),
    ('\u221a', '\\sqrt'),
    ('\u2192', '\\to'),
    ('\u2190', '\\gets'),
    ('\u2194', '\\leftrightarrow'),
    ('\u21d2', '\\Rightarrow'),
    ('\u21d0', '\\Leftarrow'),
    ('\u21d4', '\\Leftrightarrow'),
    ('\u22c5', '\\cdot'),
    ('\u2297', '\\otimes'),
    ('\u2295', '\\oplus'),
    ('\u00b1', '\\pm'),
    ('\u2213', '\\mp'),
    ('\u2032', "'"),
    ('\u2033', "''"),
    ('\u03b1', '\\alpha'),
    ('\u03b2', '\\beta'),
    ('\u03b3', '\\gamma'),
    ('\u03b4', '\\delta'),
    ('\u03b5', '\\epsilon'),
    ('\u03b6', '\\zeta'),
    ('\u03b7', '\\eta'),
    ('\u03b8', '\\theta'),
    ('\u03b9', '\\iota'),
    ('\u03ba', '\\kappa'),
    ('\u03bb', '\\lambda'),
    ('\u03bc', '\\mu'),
    ('\u03bd', '\\nu'),
    ('\u03be', '\\xi'),
    ('\u03c0', '\\pi'),
    ('\u03c1', '\\rho'),
    ('\u03c3', '\\sigma'),
    ('\u03c4', '\\tau'),
    ('\u03c5', '\\upsilon'),
    ('\u03c6', '\\phi'),
    ('\u03c7', '\\chi'),
    ('\u03c8', '\\psi'),
    ('\u03c9', '\\omega'),
    ('\u0393', '\\Gamma'),
    ('\u0394', '\\Delta'),
    ('\u0398', '\\Theta'),
    ('\u039b', '\\Lambda'),
    ('\u039e', '\\Xi'),
    ('\u03a0', '\\Pi'),
    ('\u03a3', '\\Sigma'),
    ('\u03a5', '\\Upsilon'),
    ('\u03a6', '\\Phi'),
    ('\u03a8', '\\Psi'),
    ('\u03a9', '\\Omega'),
]

def _normalize_unicode(s: str) -> str:
    for char, repl in _UNICODE_MAP:
        if char not in s:
            continue
        if repl.startswith('\\') and repl[1:].isalpha():
            s = re.sub(re.escape(char) + r'(?=[A-Za-z0-9])', lambda _, r=repl: r + ' ', s)
            s = s.replace(char, repl)
        else:
            s = s.replace(char, repl)
    return s


_SYMBOL_MAP = [
    (r'\\ne\b',             r'\\neq'),
    (r'\\le\b',             r'\\leq'),
    (r'\\ge\b',             r'\\geq'),
    (r'\\ldots\b',          r'\\dots'),
    (r'\\dotsc\b',          r'\\dots'),
    (r'\\dotso\b',          r'\\dots'),
    (r'\\dotsb\b',          r'\\cdots'),
    (r'\\cdotp\b',          r'\\cdot'),
    (r'\\varnothing\b',     r'\\emptyset'),
    (r'\\empty\b',          r'\\emptyset'),
    (r'\\operatorname\*',   r'\\operatorname'),
    (r'\\longrightarrow\b', r'\\to'),
    (r'\\longleftarrow\b',  r'\\gets'),
    (r'\\rightarrow\b',     r'\\to'),
    (r'\\leftarrow\b',      r'\\gets'),
    (r'\\Longrightarrow\b', r'\\Rightarrow'),
    (r'\\Longleftarrow\b',  r'\\Leftarrow'),
    (r'\\Longleftrightarrow\b', r'\\Leftrightarrow'),
    (r'\\longmapsto\b',     r'\\mapsto'),
    (r'\\iint\b',           r'\\int\\int'),
    (r'\\iiint\b',          r'\\int\\int\\int'),
]

def _normalize_symbols(s: str) -> str:
    for pat, repl in _SYMBOL_MAP:
        s = re.sub(pat, repl, s)
    return s


def _normalize_environments(s: str) -> str:
    s = re.sub(r'\\begin\{align\*\}',       r'\\begin{align}',    s)
    s = re.sub(r'\\end\{align\*\}',         r'\\end{align}',      s)
    s = re.sub(r'\\begin\{eqnarray\*?\}',   r'\\begin{align}',    s)
    s = re.sub(r'\\end\{eqnarray\*?\}',     r'\\end{align}',      s)
    s = re.sub(r'\\begin\{equation\*\}',    r'\\begin{equation}', s)
    s = re.sub(r'\\end\{equation\*\}',      r'\\end{equation}',   s)
    s = re.sub(r'\\begin\{smallmatrix\}',   r'\\begin{pmatrix}',  s)
    s = re.sub(r'\\end\{smallmatrix\}',     r'\\end{pmatrix}',    s)
    s = re.sub(r'\\begin\{multline\*?\}',   r'\\begin{align}',    s)
    s = re.sub(r'\\end\{multline\*?\}',     r'\\end{align}',      s)
    s = re.sub(r'\\begin\{gather\*?\}',     r'\\begin{align}',    s)
    s = re.sub(r'\\end\{gather\*?\}',       r'\\end{align}',      s)
    s = re.sub(r'\\begin\{flalign\*?\}',    r'\\begin{align}',    s)
    s = re.sub(r'\\end\{flalign\*?\}',      r'\\end{align}',      s)
    s = re.sub(r'\\begin\{alignat\*?\}\{[^}]*\}', r'\\begin{align}', s)
    s = re.sub(r'\\end\{alignat\*?\}',      r'\\end{align}',      s)
    return s


_ENV_NAMES = r'array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|cases|align|gather'

def _normalize_array_indent(s: str) -> str:
    def _strip_env(m: re.Match) -> str:
        inner = '\n'.join(line.strip() for line in m.group(2).splitlines())
        return m.group(1) + inner + m.group(3)

    return re.sub(
        r'(\\begin\{(?:' + _ENV_NAMES + r')\}[^}]*\}?)(.*?)(\\end\{(?:' + _ENV_NAMES + r')\})',
        _strip_env,
        s,
        flags=re.DOTALL,
    )


def _normalize_multiline(s: str) -> str:
    lines = [line.strip() for line in s.splitlines()]
    result, prev_blank = [], False
    for line in lines:
        blank = line == ''
        if blank and prev_blank:
            continue
        result.append(line)
        prev_blank = blank
    return '\n'.join(result)


def _normalize_whitespace(s: str) -> str:
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()


def normalize(latex: str) -> str:
    try:
        s = latex
        s = _strip_delimiters(s)
        s = _normalize_unicode(s)
        s = _normalize_over(s)
        s = _normalize_frac_variants(s)
        s = _normalize_font_commands(s)
        s = _normalize_environments(s)
        s = _normalize_symbols(s)
        s = _normalize_spacing(s)
        s = _normalize_left_right(s)
        s = _normalize_multiline(s)
        s = _normalize_array_indent(s)
        s = _normalize_whitespace(s)
        return s
    except Exception:
        return latex