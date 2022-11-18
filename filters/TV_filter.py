'''Total Variation (L1) Filter'''
import cvxpy as cp


def TV_filter(im, lamb=1):
    '''Total Variation (L1) Filter.

    Args:
        im (np.ndarray): The noisy image to be filtered.
        lamb (float): The free parameter (lambda) that determines how much to correct.

    Returns:
        clean_im (np.ndarray): The filtered image.
    '''

    X = cp.Variable(im.shape)
    shift = list(range(1, im.shape[0])) + [im.shape[0] - 1]
    dXdx = X[:, shift] - X
    dXdy = X[shift, :] - X
    objective = cp.Minimize(cp.norm(X - im, 1) + lamb * cp.norm(dXdx, 1) + lamb * cp.norm(dXdy, 1))

    prob = cp.Problem(objective)
    try:
        prob.solve(verbose=True)
    except cp.SolverError as e:
        print('[ERROR] TV Filter Failed.')
        print(e)
        exit()
    clean_im = X.value
    return clean_im
