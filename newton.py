def newtons(arr):
    alpha_cur = np.ndarray(len(arr))
    alpha_cur.fill(.1)

    maxIterations = 100
    for i in range(maxIterations):
        y = f(alpha_cur)
        y_ = f_(alpha_cur)
        print('\n\n', y, y_)
        if (abs(y_) < tol).all():
            print('abs(y_)<tol')
            break
        alpha_nxt = alpha_cur - y/y_  # Newton's computation
        alpha_cur = alpha_nxt
        print('alpha_cur: ', alpha_cur, 'alpha_nxt: ', alpha_nxt)
    return alpha_cur
