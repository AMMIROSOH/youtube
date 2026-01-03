import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_contours_from_edges(edge_img, min_len=80, max_contours=30):
    """
    Return multiple contours (not just the biggest), sorted by length desc.
    """
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if c is not None and len(c) >= min_len]
    contours = sorted(contours, key=len, reverse=True)[:max_contours]
    if not contours:
        raise ValueError("No contours found. Try lowering --min_len or adjusting Canny thresholds.")
    return [c.reshape(-1, 2) for c in contours]


def resample_closed_curve(points, n_samples=600):
    """
    Resample a closed curve to n_samples evenly spaced points along arc length.
    points: (N,2)
    returns: (n_samples,2)
    """
    pts = points.astype(np.float64)

    # Ensure closed
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(seg_lens)])
    total = s[-1]
    if total < 1e-9:
        raise ValueError("Contour is degenerate / too small.")

    t_new = np.linspace(0, total, n_samples, endpoint=False)
    out = np.zeros((n_samples, 2), dtype=np.float64)

    j = 0
    for i, tn in enumerate(t_new):
        while j + 1 < len(s) and s[j + 1] <= tn:
            j += 1
        t0, t1 = s[j], s[j + 1]
        alpha = 0.0 if t1 == t0 else (tn - t0) / (t1 - t0)
        out[i] = (1 - alpha) * pts[j] + alpha * pts[j + 1]
    return out


def fourier_coeffs(z):
    """
    Discrete Fourier coefficients for complex signal z of length M.
    Returns integer frequencies and shifted coeffs aligned with those freqs.
    """
    M = len(z)
    C = np.fft.fft(z) / M
    C_shift = np.fft.fftshift(C)
    freqs = np.fft.fftshift(np.fft.fftfreq(M, d=1.0)) * M
    freqs = freqs.astype(int)
    return freqs, C_shift


def select_top_terms(freqs, C, n_circles=10):
    """
    Pick the strongest Fourier terms by magnitude.
    Ensures DC (freq=0) is included.
    """
    mags = np.abs(C)
    idx = np.argsort(mags)[::-1][:n_circles]

    dc = np.where(freqs == 0)[0]
    if len(dc) == 1:
        dc = dc[0]
        if dc not in idx:
            idx[-1] = dc

    idx = np.array(sorted(idx, key=lambda i: freqs[i]))
    return freqs[idx], C[idx]


def epicycle_chain(terms_f, terms_c, t):
    """
    Build epicycles at time t in [0,1).
    Returns centers, radii, endpoints, final endpoint.
    """
    p = 0j
    centers, radii, endpoints = [], [], []
    for f, c in zip(terms_f, terms_c):
        centers.append(p)
        r = np.abs(c)
        radii.append(r)
        p = p + c * np.exp(2j * np.pi * f * t)
        endpoints.append(p)
    return centers, radii, endpoints, p


def normalize_all_contours(contours_xy):
    """
    Normalize all contours together so they keep their relative positions:
    - center around global mean
    - flip Y to cartesian
    - scale to fit nicely in [-1,1]
    """
    all_pts = np.vstack(contours_xy).astype(np.float64)
    mean = all_pts.mean(axis=0)
    all_pts = all_pts - mean
    all_pts[:, 1] *= -1  # flip y

    scale = max(np.max(np.abs(all_pts[:, 0])), np.max(np.abs(all_pts[:, 1])))
    if scale < 1e-9:
        raise ValueError("Normalization scale is too small.")
    all_pts /= scale

    # Split back
    out = []
    k = 0
    for c in contours_xy:
        n = len(c)
        cc = c.astype(np.float64) - mean
        cc[:, 1] *= -1
        cc /= scale
        out.append(cc)
        k += n
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Fourier epicycles draw multiple image contours beautifully."
    )
    parser.add_argument("image_path", help="Path to input image")

    parser.add_argument("--circles", type=int, default=10, help="Number of epicycles (Fourier terms)")
    parser.add_argument("--samples", type=int, default=1200, help="Resampled points per contour")
    parser.add_argument("--frames_per_contour", type=int, default=360, help="Frames used to draw each contour")

    parser.add_argument("--canny1", type=int, default=50, help="Canny threshold 1")
    parser.add_argument("--canny2", type=int, default=200, help="Canny threshold 2")
    parser.add_argument("--invert", action="store_true", help="Invert grayscale before edge detection")

    parser.add_argument("--min_len", type=int, default=80, help="Minimum contour length to keep")
    parser.add_argument("--max_contours", type=int, default=25, help="Max number of contours to draw")

    parser.add_argument("--save", default="", help="Output file: .mp4 or .gif (leave empty to just show)")

    # “Beautiful view” knobs
    parser.add_argument("--bg_alpha", type=float, default=0.08, help="Opacity for background contours (black)")
    parser.add_argument("--done_alpha", type=float, default=0.50, help="Opacity for finished strokes (black)")
    parser.add_argument("--trace_alpha", type=float, default=0.95, help="Opacity for current trace (black)")
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.invert:
        gray = 255 - gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, args.canny1, args.canny2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # cv2.imshow("asd", edges)
    # cv2.waitKey(500000)
    # cv2.destroyAllWindows()

    contours = get_contours_from_edges(edges, min_len=args.min_len, max_contours=args.max_contours)

    # Resample each contour (closed)
    contours_rs = [resample_closed_curve(c, n_samples=args.samples) for c in contours]

    # Compute the SAME mean/scale used by normalize_all_contours (so image aligns)
    all_pts = np.vstack(contours_rs).astype(np.float64)
    mean = all_pts.mean(axis=0)

    tmp = all_pts - mean
    tmp[:, 1] *= -1  # same y flip as normalization
    scale = max(np.max(np.abs(tmp[:, 0])), np.max(np.abs(tmp[:, 1])))

    # Normalize all together so the drawing matches the image layout
    contours_norm = normalize_all_contours(contours_rs)

    # Convert to complex signals and precompute Fourier terms for each contour
    Z = [c[:, 0] + 1j * c[:, 1] for c in contours_norm]
    terms_per_contour = []
    for z in Z:
        freqs, C = fourier_coeffs(z)
        tf, tc = select_top_terms(freqs, C, n_circles=args.circles)
        terms_per_contour.append((tf, tc))

    n_contours = len(Z)
    total_frames = args.frames_per_contour * n_contours

    # ---- Plot setup (clean + “beautiful”) ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")

    # --- Background original image (aligned to normalized contour coords) ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Map image corners (pixel coords) into your normalized coordinate system:
    # x' = (x - mean_x)/scale
    # y' = -(y - mean_y)/scale
    x_min = (0 - mean[0]) / scale
    x_max = (w - mean[0]) / scale
    y_top = -(0 - mean[1]) / scale        # y at top row (OpenCV y=0)
    y_bottom = -(h - mean[1]) / scale     # y at bottom row (OpenCV y=h)

    ax.imshow(
        img_rgb,
        extent=[x_min, x_max, y_bottom, y_top],
        origin="upper",   # IMPORTANT: matches OpenCV image row order
        alpha=0.18,       # tweak opacity
        zorder=0
    )


    # Pick a colormap that can give many distinct-ish colors
    cmap = plt.cm.turbo  # or: plt.cm.hsv, plt.cm.gist_rainbow
    circle_colors = cmap(np.linspace(0, 1, args.circles, endpoint=False))

    # Background faint contours (black, low opacity)
    bg_lines = []
    for z in Z:
        ln, = ax.plot(np.real(z), np.imag(z), color="black", alpha=args.bg_alpha, lw=1.0)
        bg_lines.append(ln)

    # Epicycle artists
    circle_lines = []
    arm_lines = []
    for i in range(args.circles):
        col = circle_colors[i]
        circ, = ax.plot([], [], color=col, alpha=0.65, lw=1.0)
        arm,  = ax.plot([], [], color=col, alpha=0.55, lw=1.0)
        circle_lines.append(circ)
        arm_lines.append(arm)

    # Finished strokes as one path with NaN breaks (black, low opacity)
    completed_line, = ax.plot([], [], color="black", alpha=args.done_alpha, lw=2.0)

    # Current trace (black, higher opacity)
    current_line, = ax.plot([], [], color="black", alpha=args.trace_alpha, lw=2.2)

    completed_x, completed_y = [], []
    trace_x, trace_y = [], []
    current_idx = 0

    def clear_epicycles():
        for i in range(args.circles):
            circle_lines[i].set_data([], [])
            arm_lines[i].set_data([], [])

    def update(frame):
        nonlocal current_idx, trace_x, trace_y, completed_x, completed_y

        idx = frame // args.frames_per_contour
        local = frame % args.frames_per_contour
        t = local / args.frames_per_contour

        # Switch contour: move previous trace to "completed" with a NaN break
        if idx != current_idx:
            if trace_x:
                completed_x.extend(trace_x + [np.nan])
                completed_y.extend(trace_y + [np.nan])
            trace_x, trace_y = [], []
            current_idx = idx
            clear_epicycles()

        terms_f, terms_c = terms_per_contour[idx]
        centers, radii, endpoints, p_final = epicycle_chain(terms_f, terms_c, t)

        # Update epicycles
        theta = np.linspace(0, 2 * np.pi, 128)
        for i, (c0, r, p1) in enumerate(zip(centers, radii, endpoints)):
            cx = np.real(c0) + r * np.cos(theta)
            cy = np.imag(c0) + r * np.sin(theta)
            circle_lines[i].set_data(cx, cy)
            arm_lines[i].set_data([np.real(c0), np.real(p1)], [np.imag(c0), np.imag(p1)])

        # Update traces
        trace_x.append(np.real(p_final))
        trace_y.append(np.imag(p_final))

        current_line.set_data(trace_x, trace_y)
        completed_line.set_data(completed_x, completed_y)

        return bg_lines + circle_lines + arm_lines + [completed_line, current_line]

    anim = FuncAnimation(fig, update, frames=total_frames, interval=16, blit=True)

    if args.save:
        if args.save.lower().endswith(".mp4"):
            # Needs ffmpeg installed
            anim.save(args.save, fps=60, dpi=170)
            print(f"Saved MP4 to: {args.save}")
        elif args.save.lower().endswith(".gif"):
            anim.save(args.save, writer="pillow", fps=30, dpi=170)
            print(f"Saved GIF to: {args.save}")
        else:
            raise ValueError("Output file must end with .mp4 or .gif")
    else:
        plt.show()


if __name__ == "__main__":
    main()
