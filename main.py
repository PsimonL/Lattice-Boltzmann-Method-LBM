from matplotlib import pyplot
import numpy as np
import matplotlib

plot_every = 100


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def main():
    Nx = 200  # cells of x dimension
    Ny = 50  # same byt y
    tau = .53
    Nt = 4000

    # lattice speeds na weights
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

    # init conditions
    F = np.ones((Ny, Nx, NL)) + .01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    cylinder = np.full((Ny, Nx), False)

    for y in range(0, Ny):
        for x in range(0, Nx):
            # if distance(Nx//4, Ny // 2, x, y) < 20:
            if distance(Nx // 2, Ny // 2, x, y) < 4:
                cylinder[y][x] = True

    # main loop
    for it in range(Nt):
        print(it)

        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy, in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        # Fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        F[cylinder, :] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0
        # collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)
        F = F + -(1 / tau) * (F - Feq)
        if it % plot_every == 0:
            pyplot.imshow(np.sqrt(ux ** 2 + uy ** 2), cmap = matplotlib.cm.get_cmap('plasma'))
            pyplot.pause(.01)
            pyplot.cla()



if __name__ == "__main__":
    main()