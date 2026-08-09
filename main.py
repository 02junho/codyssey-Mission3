"""main.py — Mini NPU Simulator

MAC(Multiply-Accumulate) 연산으로 입력 패턴이 어떤 필터(십자가/X)에 더 가까운지 판별한다.

핵심 아이디어
    두 개의 2차원 배열을 겹쳐 같은 위치끼리 곱하고(Multiply) 모두 더한다(Accumulate).
    점수가 높을수록 그 필터와 닮았다는 뜻이다.

실행 모드
    1) 사용자 입력 (3x3) : 필터 A/B 와 패턴을 직접 입력해 판정
    2) data.json 분석    : 5x5 / 13x13 / 25x25 필터와 패턴을 일괄 판정

외부 라이브러리를 쓰지 않고 표준 라이브러리(json, time, os)만 사용한다.
"""

import os
import json
import time

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# data.json 은 이 파일과 같은 폴더(프로젝트 루트)에 있다.
# 어느 위치에서 실행하든 같은 파일을 가리키도록 절대 경로로 계산한다.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

# 표준 라벨: 프로그램 내부에서는 항상 이 두 값만 사용한다.
LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 점수 비교 허용 오차.
# 부동소수점 연산은 0.1 + 0.2 != 0.3 처럼 미세한 오차가 남기 때문에,
# 두 점수의 차이가 이 값보다 작으면 '동점'으로 간주한다.
EPSILON = 1e-9

# 성능 측정 반복 횟수 (스펙 최소 요구: 10회)
REPEAT = 10

# 3x3 기본 필터와 패턴.
# data.json 에는 3x3 이 없지만 성능 분석 표에는 3x3 이 포함되어야 하므로 내장한다.
CROSS_3X3 = [
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
]
X_3X3 = [
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
]


# ---------------------------------------------------------------------------
# 데이터 구조: n x n 2차원 배열
# ---------------------------------------------------------------------------
def make_matrix(n, fill=0.0):
    """n x n 크기의 2차원 배열을 만들어 돌려준다."""
    return [[fill for _ in range(n)] for _ in range(n)]


def get_value(matrix, row, col):
    """특정 위치(row, col)의 값을 읽어온다."""
    return matrix[row][col]


def set_value(matrix, row, col, value):
    """특정 위치(row, col)에 값을 저장한다."""
    matrix[row][col] = value


def size_of(matrix):
    """2차원 배열의 크기 N 을 돌려준다. 정사각형이 아니면 -1."""
    rows = len(matrix)
    for row in matrix:
        if len(row) != rows:
            return -1
    return rows


# ---------------------------------------------------------------------------
# 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_label(raw):
    """데이터에 적힌 라벨을 표준 라벨(Cross / X)로 바꾼다.

    data.json 은 두 가지 표기를 섞어 쓴다.
        expected 값 : '+' 와 'x'
        filter 키   : 'cross' 와 'x'
    비교할 때마다 이 차이를 신경 쓰면 실수가 나기 때문에,
    읽어 들이는 즉시 한 가지 표기로 통일한다.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("+", "cross", "plus"):
        return LABEL_CROSS
    if text in ("x", "cross_x", "times"):
        return LABEL_X
    return None


# ---------------------------------------------------------------------------
# MAC 연산
# ---------------------------------------------------------------------------
def mac(pattern, filter_matrix):
    """MAC(Multiply-Accumulate) 연산.

    같은 위치의 값끼리 곱하고(Multiply) 그 결과를 모두 더한다(Accumulate).
    NumPy 같은 외부 라이브러리 없이 반복문으로 직접 구현한다.

    N x N 배열이면 곱셈과 덧셈을 각각 N**2 번 수행하므로 시간 복잡도는 O(N**2) 이다.
    """
    total = 0.0
    for i in range(len(pattern)):
        pattern_row = pattern[i]
        filter_row = filter_matrix[i]
        for j in range(len(pattern_row)):
            total += pattern_row[j] * filter_row[j]
    return total


def decide(score_cross, score_x):
    """두 점수를 비교해 판정 결과를 돌려준다.

    부동소수점 오차 때문에 '수학적으로 같은 값'도 미세하게 다르게 계산될 수 있다.
    그래서 단순히 > 로 비교하지 않고, 차이가 EPSILON 보다 작으면 동점으로 본다.
    동점이면 어느 쪽이라고 단정할 수 없으므로 UNDECIDED 를 돌려준다.
    """
    if abs(score_cross - score_x) < EPSILON:
        return LABEL_UNDECIDED
    return LABEL_CROSS if score_cross > score_x else LABEL_X


# ---------------------------------------------------------------------------
# 성능 측정
# ---------------------------------------------------------------------------
def measure_mac_ms(pattern, filter_matrix, repeat=REPEAT):
    """MAC 연산을 repeat 회 반복 실행하고 1회당 평균 시간(ms)을 돌려준다.

    파일 읽기나 화면 출력 시간이 섞이지 않도록,
    mac() 호출 직전과 직후만 측정한다.
    time.perf_counter() 는 시스템 시계 변경에 영향받지 않는 고해상도 타이머다.
    """
    total_seconds = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, filter_matrix)
        total_seconds += time.perf_counter() - start
    return (total_seconds / repeat) * 1000.0


def print_performance_table(entries):
    """크기별 평균 연산 시간과 연산 횟수를 표로 출력한다.

    entries: [(N, 패턴, 필터), ...]
    """
    print("\n" + "-" * 46)
    print(f"# 성능 분석 (평균 / {REPEAT}회 반복)")
    print("-" * 46)
    print(f"{'크기':<10}{'평균 시간(ms)':>16}{'연산 횟수(N^2)':>18}")
    print("-" * 46)
    for n, pattern, filter_matrix in entries:
        avg_ms = measure_mac_ms(pattern, filter_matrix)
        label = f"{n}x{n}"
        print(f"{label:<10}{avg_ms:>16.4f}{n * n:>18}")
    print("-" * 46)


# ---------------------------------------------------------------------------
# 입력 도우미
# ---------------------------------------------------------------------------
def read_int(prompt, min_value, max_value):
    """min_value ~ max_value 범위의 정수를 올바르게 입력할 때까지 반복해 받는다.

    처리하는 경우
        - 앞뒤 공백 제거 후 판단
        - 빈 입력(그냥 Enter)
        - 숫자로 바꿀 수 없는 입력(abc 등)
        - 허용 범위를 벗어난 숫자
    """
    while True:
        raw = input(prompt).strip()
        if raw == "":
            print(f"입력 오류: 값이 비어 있습니다. {min_value}~{max_value} 사이의 숫자를 입력하세요.")
            continue
        try:
            value = int(raw)
        except ValueError:
            print(f"입력 오류: 숫자가 아닙니다. {min_value}~{max_value} 사이의 숫자를 입력하세요.")
            continue
        if value < min_value or value > max_value:
            print(f"입력 오류: 범위를 벗어났습니다. {min_value}~{max_value} 사이의 숫자를 입력하세요.")
            continue
        return value


def read_matrix(title, n):
    """n 줄을 입력받아 n x n 2차원 배열을 만든다.

    각 줄은 숫자 n 개를 공백으로 구분해 입력한다.
    개수가 맞지 않거나 숫자로 바꿀 수 없으면 안내 후 그 줄을 다시 받는다.
    """
    print(f"\n{title} ({n}줄 입력, 공백 구분)")
    matrix = []
    row_index = 0
    while row_index < n:
        raw = input(f"  {row_index + 1}행: ").strip()
        if raw == "":
            print(f"  입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
            continue

        tokens = raw.split()
        if len(tokens) != n:
            print(f"  입력 형식 오류: {n}개가 필요한데 {len(tokens)}개가 입력되었습니다. 다시 입력하세요.")
            continue

        try:
            row = [float(token) for token in tokens]
        except ValueError:
            print("  입력 형식 오류: 숫자로 바꿀 수 없는 값이 있습니다. 다시 입력하세요.")
            continue

        matrix.append(row)
        row_index += 1
    return matrix


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------------------
def run_manual_mode():
    """3x3 필터 두 개와 패턴을 직접 입력받아 판정한다."""
    n = 3

    print("\n" + "=" * 46)
    print("# [1] 필터 입력")
    print("=" * 46)
    filter_a = read_matrix("필터 A", n)
    filter_b = read_matrix("필터 B", n)
    print("\n필터 A, B 저장 완료.")

    print("\n" + "=" * 46)
    print("# [2] 패턴 입력")
    print("=" * 46)
    pattern = read_matrix("패턴", n)
    print("\n패턴 저장 완료.")

    print("\n" + "=" * 46)
    print("# [3] MAC 결과")
    print("=" * 46)
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = measure_mac_ms(pattern, filter_a)

    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{REPEAT}회): {avg_ms:.4f} ms")

    verdict = decide(score_a, score_b)
    if verdict == LABEL_UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    elif verdict == LABEL_CROSS:
        print("판정: A")
    else:
        print("판정: B")

    # 스펙 요구: 사용자 입력 모드에서도 3x3 성능 분석을 출력한다.
    print_performance_table([(n, pattern, filter_a)])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_data():
    """data.json 을 읽어 딕셔너리로 돌려준다. 실패하면 None."""
    if not os.path.exists(DATA_FILE):
        print(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
        return None
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("데이터 파일이 손상되어 JSON 파싱에 실패했습니다.")
        return None
    except OSError:
        print("데이터 파일을 읽는 중 오류가 발생했습니다.")
        return None


def load_filters(data):
    """filters 를 {N: {Cross: 배열, X: 배열}} 형태로 정규화해 돌려준다.

    'size_5' 같은 키에서 숫자 5 를 뽑아내고,
    'cross' / 'x' 키는 표준 라벨(Cross / X)로 바꾼다.
    """
    print("\n" + "=" * 46)
    print("# [1] 필터 로드")
    print("=" * 46)

    filters = {}
    raw_filters = data.get("filters", {})
    for key in sorted(raw_filters, key=lambda k: parse_size(k) or 0):
        n = parse_size(key)
        if n is None:
            print(f"  [건너뜀] {key}: 필터 키에서 크기를 읽을 수 없습니다.")
            continue

        normalized = {}
        for label_key, matrix in raw_filters[key].items():
            label = normalize_label(label_key)
            if label is None:
                print(f"  [건너뜀] {key}.{label_key}: 알 수 없는 라벨입니다.")
                continue
            normalized[label] = matrix

        if LABEL_CROSS in normalized and LABEL_X in normalized:
            filters[n] = normalized
            print(f"  [OK] {key} 필터 로드 완료 ({LABEL_CROSS}, {LABEL_X})")
        else:
            print(f"  [경고] {key}: Cross/X 필터가 모두 있어야 합니다.")
    return filters


def parse_size(key):
    """'size_5' 또는 'size_5_1' 같은 키에서 크기 N 을 뽑아낸다. 실패하면 None."""
    parts = str(key).split("_")
    if len(parts) < 2 or parts[0] != "size":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def analyze_patterns(data, filters):
    """patterns 를 하나씩 판정하고 결과 목록을 돌려준다.

    한 케이스에서 문제가 생겨도 프로그램을 멈추지 않고
    그 케이스만 FAIL 로 기록한 뒤 다음으로 넘어간다.
    """
    print("\n" + "=" * 46)
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("=" * 46)

    results = []
    for key, item in data.get("patterns", {}).items():
        result = {"case": key, "passed": False, "reason": ""}

        # (1) 키에서 크기 N 추출
        n = parse_size(key)
        if n is None:
            result["reason"] = "패턴 키에서 크기를 읽을 수 없음"
            print(f"\n--- {key} ---\n  FAIL ({result['reason']})")
            results.append(result)
            continue

        # (2) 해당 크기의 필터가 있는지
        if n not in filters:
            result["reason"] = f"size_{n} 필터가 없음"
            print(f"\n--- {key} ---\n  FAIL ({result['reason']})")
            results.append(result)
            continue

        pattern = item.get("input")
        expected = normalize_label(item.get("expected"))

        # (3) 패턴이 정사각 2차원 배열인지
        if not isinstance(pattern, list) or size_of(pattern) == -1:
            result["reason"] = "패턴 input 이 정사각 2차원 배열이 아님"
            print(f"\n--- {key} ---\n  FAIL ({result['reason']})")
            results.append(result)
            continue

        # (4) 패턴과 필터의 크기가 일치하는지
        pattern_size = size_of(pattern)
        filter_size = size_of(filters[n][LABEL_CROSS])
        if pattern_size != n or filter_size != n:
            result["reason"] = (
                f"크기 불일치 (키={n}, 패턴={pattern_size}, 필터={filter_size})"
            )
            print(f"\n--- {key} ---\n  FAIL ({result['reason']})")
            results.append(result)
            continue

        # (5) expected 라벨이 유효한지
        if expected is None:
            result["reason"] = f"expected 값을 표준 라벨로 바꿀 수 없음: {item.get('expected')!r}"
            print(f"\n--- {key} ---\n  FAIL ({result['reason']})")
            results.append(result)
            continue

        # (6) MAC 연산과 판정
        score_cross = mac(pattern, filters[n][LABEL_CROSS])
        score_x = mac(pattern, filters[n][LABEL_X])
        verdict = decide(score_cross, score_x)

        passed = (verdict == expected)
        result["passed"] = passed
        if not passed:
            if verdict == LABEL_UNDECIDED:
                result["reason"] = f"동점(UNDECIDED) 처리 규칙에 따라 판정 보류 (expected={expected})"
            else:
                result["reason"] = f"판정({verdict})이 expected({expected})와 다름"

        print(f"\n--- {key} ---")
        print(f"  {LABEL_CROSS} 점수: {score_cross!r}")
        print(f"  {LABEL_X} 점수: {score_x!r}")
        print(f"  판정: {verdict} | expected: {expected} | {'PASS' if passed else 'FAIL'}")

        results.append(result)
    return results


def print_summary(results):
    """전체 / 통과 / 실패 수와 실패 케이스 목록을 출력한다."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 46)
    print("# [4] 결과 요약")
    print("=" * 46)
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")

    if failed:
        print("\n실패 케이스:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['case']}: {r['reason']}")
    else:
        print("\n실패한 케이스가 없습니다.")


def run_json_mode():
    """data.json 을 읽어 전체 패턴을 판정하고 성능 분석까지 출력한다."""
    data = load_data()
    if data is None:
        print("데이터를 불러오지 못해 분석을 중단합니다.")
        return

    filters = load_filters(data)
    if not filters:
        print("사용 가능한 필터가 없어 분석을 중단합니다.")
        return

    results = analyze_patterns(data, filters)

    # 성능 분석: 3x3 은 내장 데이터로, 나머지는 data.json 의 실제 필터/패턴으로 측정한다.
    # 같은 크기를 두 번 재지 않도록 이미 넣은 크기는 건너뛴다.
    entries = [(3, X_3X3, CROSS_3X3)]
    measured = {3}
    for n in sorted(filters):
        if n in measured:
            continue
        sample = find_pattern_of_size(data, n)
        if sample is not None:
            entries.append((n, sample, filters[n][LABEL_CROSS]))
            measured.add(n)
    print_performance_table(entries)

    print_summary(results)


def find_pattern_of_size(data, n):
    """크기가 n 인 패턴 하나를 찾아 돌려준다. 없으면 None."""
    for key, item in data.get("patterns", {}).items():
        if parse_size(key) != n:
            continue
        pattern = item.get("input")
        if isinstance(pattern, list) and size_of(pattern) == n:
            return pattern
    return None


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def show_menu():
    """모드 선택 메뉴를 출력하고 고른 번호를 돌려준다."""
    print("\n" + "=" * 46)
    print("        Mini NPU Simulator")
    print("=" * 46)
    print("[모드 선택]")
    print("  1. 사용자 입력 (3x3)")
    print("  2. data.json 분석")
    print("  3. 종료")
    print("=" * 46)
    return read_int("선택: ", 1, 3)


def main():
    try:
        while True:
            choice = show_menu()
            if choice == 1:
                run_manual_mode()
            elif choice == 2:
                run_json_mode()
            else:
                print("\n프로그램을 종료합니다.")
                break
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C 나 입력 종료로도 비정상 종료되지 않도록 처리한다.
        print("\n\n입력이 중단되어 안전하게 종료합니다.")


if __name__ == "__main__":
    main()
