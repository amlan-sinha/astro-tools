import numpy as np

DAYS2SECONDS=86400
SECONDS2DAYS=1./86400


class Kepler:

    def __init__(self, central_body):

        self.radius, self.mu, self.sma = central_body.radius, central_body.mu, central_body.sma

    def sphere_of_influence(self, target_body):

        radius, mu, sma = target_body.radius, target_body.mu, target_body.sma

        return sma * (mu / self.mu)**(2 / 5)

    def hohmann(self, ri, rf):

        mu, re = self.mu, self.radius

        ri, rf = ri + re, rf + re
        atrans = 0.5 * (ri + rf)
        vi, vf = np.sqrt(mu / ri), np.sqrt(mu / rf)
        vtrans_a, vtrans_b = np.sqrt(2 * mu / ri - mu / atrans), np.sqrt(2 * mu / rf - mu / atrans)
        dv_a, dv_b = vtrans_a - vi, vf - vtrans_b

        dv, tof = np.abs(dv_a) + np.abs(dv_b), np.pi * np.sqrt(atrans**3 / mu) * SECONDS2DAYS

        return (dv, tof)

    def bielliptic(self, ri, rb, rf):

        mu, re = self.mu, self.radius

        ri, rb, rf = ri + re, rb + re, rf + re
        atrans1, atrans2 = 0.5 * (ri + rb), 0.5 * (rb + rf)
        vi, vf = np.sqrt(mu / ri), np.sqrt(mu / rf)
        vtrans1_a, vtrans1_b = np.sqrt(2 * mu / ri - mu / atrans1), np.sqrt(2 * mu / rb - mu / atrans1)
        vtrans2_b, vtrans2_c = np.sqrt(2 * mu / rb - mu / atrans2), np.sqrt(2 * mu / rf - mu / atrans2)
        dv_a, dv_b, dv_c = vtrans1_a - vi, vtrans2_b - vtrans1_b, vf - vtrans2_c

        dv, tof = np.abs(dv_a) + np.abs(dv_b) + np.abs(dv_c), np.pi * ( np.sqrt(atrans1**3 / mu) + np.sqrt(atrans2**3 / mu) ) * SECONDS2DAYS

        return (dv, tof)

    def raise_apogee(self, rp, rai, raf):

        mu, re = self.mu, self.radius

        rp, rai, raf = rp + re, rai + re, raf + re
        ai, af = 0.5 * (rp + rai), 0.5 * (rp + raf)
        vi, vf = np.sqrt(2 * mu / rp - mu / ai), np.sqrt(2 * mu / rp - mu / af)
        dv = vf - vi

        dv, tof = np.abs(dv), 0. * SECONDS2DAYS

        return (dv, tof)

    # def compare_hohmann_vs_bielliptic():

    #     def dv_bielliptic_effective(Rs, R):
    #         return np.abs(np.sqrt(2 * Rs / (1 + Rs)) - 1) + \
    #             np.abs(np.sqrt(2 / Rs) * (np.sqrt(1 / (1 + Rs / R)) - np.sqrt(1 / (1 + Rs)))) + \
    #             np.abs(np.sqrt(1 / R) * (np.sqrt(2 * Rs / (R + Rs)) - 1))

    #     R = np.logspace(-1.5, 2, 1000) # R = r_f / r_i
    #     Rsvec = np.array([2, 3, 4, 5, 15.58171, 50, 100, 500]) # R = r_b / r_i

    #     dv_hohmann_effective = np.abs((1 - 1 / R) * np.sqrt(2 * R / (1 + R)) + np.sqrt(1 / R) - 1)
    #     dv_inf_effective = np.abs((np.sqrt(2) - 1) * (1 + np.sqrt(1 / R)))

    #     fig, ax = plt.subplots(figsize=(15, 15))
    #     ax.plot(R, dv_hohmann_effective, c='k', linewidth=3, label="Hohmann")
    #     for rsi in Rsvec:
    #         this_label = r"$R_s$ = {:.3f}".format(rsi)
    #         ax.plot(R, dv_bielliptic_effective(rsi, R), linewidth=1, label=this_label)
    #     ax.plot(R, dv_inf_effective, c='k', linestyle='--', linewidth=3, label=r"$R_s = \infty$")
    #     ax.vlines(11.94, np.amin(dv_hohmann_effective), np.amax(dv_hohmann_effective), linestyles ="dashed", colors ="k")
    #     ax.text(11.94, 0.75 * (np.amin(dv_hohmann_effective) + np.amax(dv_hohmann_effective)), "R = 11.94", horizontalalignment='center', c='k', fontsize=20)
    #     ax.vlines(15.58, np.amin(dv_hohmann_effective), np.amax(dv_hohmann_effective), linestyles ="dashed", colors ="k")
    #     ax.text(15.58, 0.5 * (np.amin(dv_hohmann_effective) + np.amax(dv_hohmann_effective)), "R = 15.58", horizontalalignment='center', c='k', fontsize=20)
    #     ax.set_xlabel(r"$R = \frac{r_i}{r_f}$", fontsize=50)
    #     ax.set_ylabel(r"$\Delta v / v_i$", fontsize=50)
    #     ax.set_xlim([1 / 11.94, np.amax(R)])
    #     ax.set_ylim([0.25, np.amax(dv_hohmann_effective)])
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.tick_params(axis='both', which='major', labelsize=25)
    #     ax.legend(fontsize=15, loc='upper left')

    #     fig.savefig("hohmann_vs_bielliptic.png", dpi=300)


class ConicSections:

    def __init__(self, radius, rp, ra):

        self.rp = rp + radius
        self.ra = ra + radius
        self.sma = 0.5 * (self.rp + self.ra)
