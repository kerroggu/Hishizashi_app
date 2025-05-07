import cv2
import numpy as np
from pathlib import Path

# ------------------------------------------------------
# diamond_pattern_extractor  v1.2
#   - ひし形パターンを 0/1 配列に変換
#   - 自動回転が起きないように長辺を必ず横方向へ揃える
#   - --show オプションで白黒プレビュー
# ------------------------------------------------------

PATTERN_W, PATTERN_H = 34, 17  # ドット数（横×縦）

# ---- I/O ---------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img

# ---- マスク作成 --------------------------------------------------------

def extract_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 40, 0])     # 紺色域（要調整）
    upper = np.array([135, 255, 200])
    mask = cv2.inRange(hsv, lower, upper)

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

# ---- ラベリング --------------------------------------------------------

def get_components(mask: np.ndarray, min_area: int = 200):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, num):  # 0 は背景
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp_mask = (labels == i).astype(np.uint8) * 255
        comps.append((comp_mask, centroids[i]))
    return comps

# ---- 射影変換（回転固定版） -------------------------------------------

def rectify_component(mask: np.ndarray) -> np.ndarray:
    """コンポーネントを外接 axis-aligned bbox で切り出して返す"""
    y_idxs, x_idxs = np.where(mask > 0)
    y0, y1 = y_idxs.min(), y_idxs.max()
    x0, x1 = x_idxs.min(), x_idxs.max()
    roi = mask[y0:y1 + 1, x0:x1 + 1]
    return roi


# ---- 0/1 配列化 --------------------------------------------------------

def binarize(warped: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(warped, 127, 1, cv2.THRESH_BINARY)
    return bw.astype(np.uint8)

# ---- 可視化 -----------------------------------------------------------

def visualize_array(arr: np.ndarray, scale: int = 10) -> np.ndarray:
    img = (1 - arr) * 255  # 背景=白, 模様=黒
    img = cv2.resize(img, (arr.shape[1] * scale, arr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return img

def majority_sample(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    cell_h = h / PATTERN_H
    cell_w = w / PATTERN_W
    grid = np.zeros((PATTERN_H, PATTERN_W), np.uint8)
    for r in range(PATTERN_H):
        y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
        for c in range(PATTERN_W):
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            cell = mask[y0:y1, x0:x1]
            grid[r, c] = (cell.mean() > 127)
    return grid * 255  # binarize() 用 0/255
# ---- メイン -----------------------------------------------------------

def extract_pattern_arrays(image_path: str, out_dir: str = "patterns", visualize: bool = False):
    img = load_image(image_path)
    mask = extract_mask(img)
    comps = get_components(mask)

    # 上→下, 左→右 でソート
    comps.sort(key=lambda c: (c[1][1], c[1][0]))

    Path(out_dir).mkdir(exist_ok=True)
    outputs = []
    for idx, (comp_mask, centroid) in enumerate(comps):
        warped = rectify_component(comp_mask)
        
        
        warped=majority_sample(warped)
        N=0
        for i in range(N)[::-1]:
            warped = cv2.resize(warped, (PATTERN_W*(2**i), PATTERN_H*(2**i)), interpolation=cv2.INTER_NEAREST)
        arr = binarize(warped)
        np.save(Path(out_dir) / f"pattern_{idx:02d}.npy", arr)
        outputs.append(arr)

        if visualize:
            cv2.imshow(f"pattern_{idx:02d}", visualize_array(arr))

    if visualize:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return outputs

# ---- CLI --------------------------------------------------------------
if __name__ == "__main__":

    import argparse
    # ★ Jupyter用：先にsys.argvを上書き！
    img = "Screenshot_20250506-132354.png"
    img = "2.png"
    out = "PyOut"
    sys.argv = ['notebook', img, '--out', out, '--show']

    
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="input image file")
    parser.add_argument("--out", default="patterns", help="output directory")
    parser.add_argument("--show", action="store_true", help="show pattern preview windows")
    args = parser.parse_args()

    extract_pattern_arrays(args.image, args.out, args.show)
