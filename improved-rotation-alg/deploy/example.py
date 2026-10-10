"""Minimal, self-contained example. Dependency: numpy only.

    python example.py

Builds a synthetic gel with a KNOWN tilt, runs the estimator, and prints the result - so the
integration can be checked without any sample data or segmentation model. For real use the input
is whatever band centroids your segmenter produced (see README.md).
"""
import numpy as np
import ladders

LADDER_RUNGS = 8          # the ladder lane: a dense, evenly spaced size series
TILT_DEG = 2.0            # the tilt the lanes actually have
NOISE_X = 0.35            # px, band centroid localisation error in x
NOISE_Y = 0.80            # px, in y


def synthetic_gel(tilt_deg=TILT_DEG, n_lanes=6, seed=0):
    """A ladder lane plus sample lanes of 7-9 regularly spaced bands.

    The rung counts matter: a real lane - ladder or loaded sample - has many bands in a regular
    series. Two or three bands at random heights are not a lane and the estimator will (correctly)
    not fit a line to them. Returns (xy, truth_deg).
    """
    rng = np.random.default_rng(seed)
    slope = -np.tan(np.radians(tilt_deg))          # dx/dy for a lane tilted by tilt_deg
    pts = []
    for lane in range(n_lanes):
        x0 = 90.0 + lane * 190.0
        if lane == 0:
            ys = np.linspace(100.0, 760.0, LADDER_RUNGS)
        else:
            n = int(rng.integers(7, 10))
            top = float(rng.uniform(90.0, 170.0))
            pitch = float(rng.uniform(55.0, 85.0))
            ys = np.cumsum(rng.normal(pitch, pitch * 0.06, n)) + top
        for y in ys:
            pts.append([x0 + slope * (y - 430.0) + rng.normal(0, NOISE_X),
                        y + rng.normal(0, NOISE_Y)])
    # the angle to APPLY that makes the lanes vertical is the tilt itself
    return np.array(pts, float), tilt_deg


def main():
    xy, truth = synthetic_gel()
    print('input: %d band centroids, %d columns, x %.0f..%.0f, y %.0f..%.0f'
          % (len(xy), len(np.unique(np.round(xy[:, 0] / 100))), xy[:, 0].min(), xy[:, 0].max(),
             xy[:, 1].min(), xy[:, 1].max()))

    res = ladders.estimate(xy)

    if not res['ok']:
        print('refused: %s' % res['reason'])
        return
    print('\nangle to apply : %+.3f deg   (truth %+.3f, error %+.3f)'
          % (res['angle'], truth, res['angle'] - truth))
    print('uncertainty    : %.3f deg  (se; |error| should be well under this on average)'
          % res['se'])
    print('lanes          : %d found, %d pooled, %d rungs'
          % (res['n_ladders'], res['kept'], res['n_rungs']))
    print('overdispersion : %.2f   (1.0 = the lanes agree with their own errors)' % res['overdispersion'])
    if res['warnings']:
        print('warnings       :')
        for w in res['warnings']:
            print('                 - %s' % w)

    print('\nper-lane detail (ranked by trustworthiness):')
    print('  %-4s %5s %7s %8s %9s %8s %9s %6s'
          % ('lane', 'rungs', 'span', 'pitch', 'consist', 'angle', 'se_deg', 'trust'))
    for i, l in enumerate(res['ladders']):
        mark = '' if i < res['kept'] else '  (not pooled)'
        print('  %-4d %5d %7.0f %8.1f %9.2f %+8.3f %9.3f %6.1f%s'
              % (i, l['n'], l['span'], l['pitch'], l['consistency'], l['angle'],
                 l['se_deg_cal'], l['trust'], mark))

    # how a host application applies it
    print('\napply with OpenCV:')
    print('    M = cv2.getRotationMatrix2D((W / 2, H / 2), %.3f, 1.0)' % res['angle'])
    print('    corrected = cv2.warpAffine(image, M, (W, H))')


if __name__ == '__main__':
    main()
