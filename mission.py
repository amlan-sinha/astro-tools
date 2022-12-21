import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os, datetime

cmap = plt.get_cmap('bwr')
mycolors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

DAYS2SECONDS=86400
SECONDS2DAYS=1./86400

from keplerian.tools import Kepler, ConicSections
from keplerian.Bodies import Earth, Moon

class Mission():

    def __init__(self, if_lowspeed_reentry=True, if_margin=True):

        earth, moon = Earth(), Moon()

        self.system, self.other = Kepler(earth), Kepler(moon)
        self.earth_radius, self.moon_radius = earth.equatorial_radius, moon.equatorial_radius
        self.earth_mu, self.moon_mu = earth.mu, moon.mu

        self.initial_altitude, self.terminal_altitude = 500, 100 # km
        self.initial_gto_perigee_altitude, self.initial_gto_apogee_altitude = 200, 35975 # km
        initial_perigee, initial_apogee = self.initial_gto_perigee_altitude + self.earth_radius, self.initial_gto_apogee_altitude + self.earth_radius # km
        self.initial_sma, self.initial_eccentricity = 0.5 * (initial_perigee + initial_apogee), (initial_apogee - initial_perigee) / (initial_apogee + initial_perigee) # km

        
        self.lowspeed_reentry = if_lowspeed_reentry
        self.margin = 0.2 if if_margin else 0.
        self.g0, self.isp = 9.81, 330, # m/s^2, s
        self.initial_mass, self.example_lunar_lander_mass = 45, (2445 + 2034) * 1e-3 # mT
        self.earth_reentry_altitude = 150 # km
        self.minimum_reentry_speed = 7.8 # km/s
        
        self.distance = moon.semimajor_axis # km
        self.velocity = np.sqrt(moon.mu_of_parent / moon.semimajor_axis) # km / s

        self.path_to_results_dir = "./results/" + str(datetime.datetime.now().date())
        if not os.path.isdir(self.path_to_results_dir):
            os.makedirs(self.path_to_results_dir)
        self.path_to_figure_dir = self.path_to_results_dir + "/figures"
        if not os.path.isdir(self.path_to_figure_dir):
            os.makedirs(self.path_to_figure_dir)

    def switch_central_body(self):
        """ 
        
        Helper function for switching central body

        """

        tmp = self.system
        self.system = self.other
        self.other = tmp

    def get_propellant_mass_fraction(self, dv):
        """ 
        
        Helper function for getting propellant mass fraction

        """
        return 1 - np.exp(-dv / self.isp / self.g0)

    def get_mass_fraction(self, dv):
        """ 
        
        Helper function for getting mass fraction

        """
        return np.exp(-dv / self.isp / self.g0)

    def get_delta_v(self, control, number_of_segments) -> float:
        """ 
        
        Helper function for calculating the delta-v from a control output

        """
        delta_v = 0.
        for i in range(number_of_segments):
            starting_index = i * 3
            ending_index = (i + 1) * 3
            delta_v += abs(np.linalg.norm(control[starting_index:ending_index]))

        return delta_v

    def output_control_to_screen(self, control):
        """
        
        Helper function for outputing control to stdout

        """
        print('\nOptimal Control Input:')
        for entry in control:
            print('{},'.format(entry))

    def calculate_delta_v_loi(self, initial_orbit, terminal_altitude):
        """ 
        
        Calculate the delta-v and the transfer time necessary to perform LOI


        Parameters
        ----------
        initial_orbit : ConicSection
            Initial orbit
            
        terminal_altitude : float
            Altitude of the LLO
        

        Returns
        -------
        tuple
            (delta-v, time-of-flight)

        """

        theta = 0. # assuming that the angle between the transfer orbit and the moon's orbit is zero
        terminal_altitude += self.moon_radius

        v_moon_rel_earth = self.velocity  # mean orbital velocity of the moon in ECI
        v_sc_rel_earth = np.sqrt(2 * self.earth_mu / initial_orbit.ra - self.earth_mu / initial_orbit.sma)  # orbital velocity of the spacecraft in ECI
        v_sc_inf = np.sqrt(v_moon_rel_earth**2 + v_sc_rel_earth**2 - 2 * v_moon_rel_earth * v_sc_rel_earth * np.cos(theta))  # excess velocity of the spacecraft in MCI
        v_sc_rel_moon = np.sqrt(v_sc_inf**2 + 2 * self.moon_mu / terminal_altitude)  # spacecraft orbital velocity in LCI
        
        return np.abs(v_sc_rel_moon - np.sqrt(self.moon_mu / terminal_altitude)), np.pi * np.sqrt(initial_orbit.sma**3 / self.earth_mu) * SECONDS2DAYS

    def calculate_delta_v_tei(self, initial_altitude, terminal_orbit):
        """ 
        
        Calculate the delta-v and the transfer time necessary to perform TEI


        Parameters
        ----------            
        initial_altitude : float
            Altitude of the LLO

        terminal_orbit : ConicSection
            Terminal orbit
        

        Returns
        -------
        tuple
            (delta-v, time-of-flight)

        """

        theta = 0. # assuming that the angle between the transfer orbit and the moon's orbit is zero
        initial_altitude += self.moon_radius

        v_moon_rel_earth = self.velocity  # mean orbital velocity of the moon in ECI
        v_sc_rel_earth = np.sqrt(2 * self.earth_mu / terminal_orbit.ra - self.earth_mu / terminal_orbit.sma)  # orbital velocity of the spacecraft in ECI
        v_sc_inf = np.sqrt(v_moon_rel_earth**2 + v_sc_rel_earth**2 - 2 * v_moon_rel_earth * v_sc_rel_earth * np.cos(theta))  # excess velocity of the spacecraft in MCI
        v_sc_rel_moon = np.sqrt(v_sc_inf**2 + 2 * self.moon_mu / initial_altitude)  # spacecraft orbital velocity in LCI
        
        return np.abs(v_sc_rel_moon - np.sqrt(self.moon_mu / initial_altitude)), np.pi * np.sqrt(terminal_orbit.sma**3 / self.earth_mu) * SECONDS2DAYS

    def LEO_to_LLO(self):
        """

        Calculate the delta-v and the transfer times for LEO to LLO

        We consider a circular parking orbit around Earth (e.g., LEO) as our initial condition, and aim to insert ourselves into a circular orbit around the Moon.
        We assume that we have access to a capsule which detaches from the spacecraft, and descends to the lunar surface.

        Parameters
        ----------
        if_margin : Boolean
            Whether or not we apply a margin to the delta-v at each stage

        Returns
        -------
        table
            Table of results

        """
        initial_altitude, terminal_altitude = self.initial_altitude, self.terminal_altitude

        leo = ConicSections(radius=self.earth_radius, rp=initial_altitude, ra=initial_altitude) # circular
        final_orbit = ConicSections(radius=self.earth_radius, rp=self.distance - self.earth_radius + self.moon_radius + terminal_altitude, ra=self.distance - self.earth_radius + self.moon_radius + terminal_altitude) # circular
        tei_transfer_orbit = ConicSections(radius=self.earth_radius, rp=self.initial_gto_perigee_altitude, ra=self.distance - self.earth_radius + self.moon_radius + terminal_altitude)

        # TLI
        dv_tli, tof_tli = self.system.hohmann(ri=leo.rp, rf=final_orbit.ra)
        # LOI
        dv_loi, _ = self.calculate_delta_v_loi(final_orbit, self.terminal_altitude)
        tof_loi = 0.
        # {LD, LA and EOI}
        self.switch_central_body() # now we are in the MCI frame
        # LD
        dv_ld, _ = self.system.hohmann(ri=terminal_altitude, rf=0) # from the LLO to the surface of the moon
        dv_ld += np.sqrt(self.system.mu / self.system.radius) # excess velocity at the surface of the moon
        # LA
        dv_la, _ = self.system.hohmann(ri=0, rf=terminal_altitude) # from the surface to the LLO
        dv_la += np.sqrt(self.system.mu / self.system.radius) # excess velocity at the surface of the moon
        # {TEI, ALM, PLM}
        self.switch_central_body() # now we are back in the ECI frame
        # TEI
        dv_tei, tof_tei = self.calculate_delta_v_tei(terminal_altitude, tei_transfer_orbit)
        # ALM
        dv_alm, tof_alm =  self.system.raise_apogee(rp=self.initial_gto_perigee_altitude, rai=self.distance - self.earth_radius + self.moon_radius + terminal_altitude, raf=self.initial_gto_apogee_altitude)
        # PLM
        dv_plm, tof_plm =  self.system.lower_perigee(ra=self.initial_gto_apogee_altitude, rpi=self.initial_gto_perigee_altitude, rpf=self.earth_reentry_altitude)
        if self.lowspeed_reentry:
            # LSR
            speed_in_final_reentry_orbit = np.sqrt(2 * self.earth_mu / (self.earth_reentry_altitude + self.earth_radius) - self.earth_mu / (0.5 * (self.earth_reentry_altitude + self.initial_gto_apogee_altitude + 2 * self.earth_radius)))
            dv_lsr = speed_in_final_reentry_orbit - self.minimum_reentry_speed
        # ED = ALM + PLM
        if self.lowspeed_reentry:
            dv_ed = dv_alm + dv_plm + dv_lsr
        else:
            dv_ed = dv_alm + dv_plm
        tof_ed = tof_alm + tof_plm

        dv_tot = dv_tli + dv_loi + dv_ld + dv_la + dv_tei + dv_ed
        tof_tot = tof_tli + tof_tei + tof_ed

        if self.lowspeed_reentry:
            dv_vec = np.array([dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_lsr, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot])
        else:
            dv_vec = np.array([dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot])
        lam_vec = self.get_mass_fraction((1 + self.margin) * dv_vec * 1e3)
        final_mass_at_the_end_the_stage, propellant_mass_consumed = np.zeros_like(lam_vec), np.zeros_like(lam_vec)
        for i, lam_i in enumerate(lam_vec):
            if (i==0):
                final_mass_at_the_end_the_stage[i] = lam_i * self.initial_mass
                propellant_mass_consumed[i] = (1 - lam_i) * self.initial_mass
            else:
                final_mass_at_the_end_the_stage[i] = lam_i * final_mass_at_the_end_the_stage[i - 1]
                propellant_mass_consumed[i] = (1 - lam_i) * final_mass_at_the_end_the_stage[i - 1]
        # Overwriting the parameters for the entire mission
        final_mass_at_the_end_the_stage[-3] = self.get_mass_fraction(dv_vec[-3] * 1e3) * self.initial_mass
        propellant_mass_consumed[-3] = self.get_propellant_mass_fraction(dv_vec[-3] * 1e3) * self.initial_mass
        # Overwriting the parameters for the margin entry
        final_mass_at_the_end_the_stage[-2] = 0.
        propellant_mass_consumed[-2] = 0.
        # Overwriting the parameters for the entire mission with margin
        final_mass_at_the_end_the_stage[-1] = self.get_mass_fraction(dv_vec[-1] * 1e3) * self.initial_mass
        propellant_mass_consumed[-1] = self.get_propellant_mass_fraction(dv_vec[-1] * 1e3) * self.initial_mass

        # Calculating mass fractions
        # LD
        mi_ld = np.linspace(0., final_mass_at_the_end_the_stage[1], 1000)
        mp_ld = (1 - lam_vec[2]) * mi_ld
        mf_ld = mi_ld - mp_ld
        # LA
        mi_la = mf_ld
        mp_la = (1 - lam_vec[3]) * mi_la
        mf_la = mi_la - mp_la
        # TEI
        mi_tei = final_mass_at_the_end_the_stage[1] - mi_ld + mf_la
        mp_tei = (1 - lam_vec[4]) * mi_tei
        mf_tei = mi_tei - mp_tei
        # PLM
        mi_alm = mf_tei
        mp_alm = (1 - lam_vec[5]) * mi_alm
        mf_alm = mi_alm - mp_alm
        # ALM
        mi_plm = mf_alm
        mp_plm = (1 - lam_vec[6]) * mi_plm
        mf_plm = mi_plm - mp_plm
        # LSR
        if self.lowspeed_reentry:
            mi_lsr = mf_plm
            mp_lsr = (1 - lam_vec[7]) * mi_plm
            mf_lsr = mi_lsr - mp_lsr

        print("=========================================================================================")
        if self.lowspeed_reentry:
            phase_to_print = ['TLI', 'LOI', 'LD', 'LA', 'TEI', 'ALM', 'PLM', 'LSR', 'TOTAL', 'MARGIN', 'ADJUSTED TOTAL']
            dv_to_print = [dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_lsr, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot]
            tof_to_print = [tof_tli, tof_loi, r'$\Delta_1$', r'$\Delta_2$', tof_tei, tof_alm, tof_plm, '', tof_tot, '', tof_tot]
            mi_to_print = [self.initial_mass, final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], final_mass_at_the_end_the_stage[6], self.initial_mass, '', self.initial_mass]
            mp_to_print = [propellant_mass_consumed[0], propellant_mass_consumed[1], propellant_mass_consumed[2], propellant_mass_consumed[3], propellant_mass_consumed[4], propellant_mass_consumed[5], propellant_mass_consumed[6], propellant_mass_consumed[7], propellant_mass_consumed[-3], '', propellant_mass_consumed[-1]]
            mf_to_print = [final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], final_mass_at_the_end_the_stage[6], final_mass_at_the_end_the_stage[7], final_mass_at_the_end_the_stage[-3], '', final_mass_at_the_end_the_stage[-1]]
            if (self.margin > 0):
                print("LEO to LLO Budget For Low-Speed With Margin")
            else:
                print("LEO to LLO Budget For Low-Speed Without Margin")
        else:
            phase_to_print = ['TLI', 'LOI', 'LD', 'LA', 'TEI', 'ALM', 'PLM', 'TOTAL', 'MARGIN', 'ADJUSTED TOTAL']
            dv_to_print = [dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot]
            tof_to_print = [tof_tli, tof_loi, r'$\Delta_1$', r'$\Delta_2$', tof_tei, tof_alm, tof_plm, tof_tot, '', tof_tot]
            mi_to_print = [self.initial_mass, final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], self.initial_mass, '', self.initial_mass]
            mp_to_print = [propellant_mass_consumed[0], propellant_mass_consumed[1], propellant_mass_consumed[2], propellant_mass_consumed[3], propellant_mass_consumed[4], propellant_mass_consumed[5], propellant_mass_consumed[6], propellant_mass_consumed[-3], '', propellant_mass_consumed[-1]]
            mf_to_print = [final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], final_mass_at_the_end_the_stage[6], final_mass_at_the_end_the_stage[-3], '', final_mass_at_the_end_the_stage[-1]]
            if (self.margin > 0):
                print("LEO to LLO Budget For High-Speed With Margin")
            else:
                print("LEO to LLO Budget For High-Speed Without Margin")
        print("=========================================================================================")
        results = {'Type': phase_to_print, 
                   'Delta-V (km/s)': dv_to_print, 
                   'Transfer Time (days)': tof_to_print,
                   'Mass At the Start Of Phase (mT)': mi_to_print,
                   'Propellant Mass Consumed (mT)': mp_to_print,
                   'Mass At the End Of Phase (mT)': mf_to_print}
        results_table = tabulate(results, headers='keys')
        print(results_table)
        # print("=========================================================================================")
        plot_title = "LEO"
        if self.lowspeed_reentry:
            plot_title += " low-speed"
        else:
            plot_title += " high-speed"
        if (self.margin > 0):
            plot_title += " with margin"
        else:
            plot_title += " without margin"
        fig_p, ax_p = plt.subplots(figsize=(16, 10))
        ax_p.set_prop_cycle(color=mycolors)
        ax_p.plot(mi_ld, mp_ld, label=r"$m_p^{LD}$", linestyle="dashed")
        ax_p.plot(mi_ld, mp_la, label=r"$m_p^{LA}$", linestyle="dashed")
        ax_p.plot(mi_ld, mp_tei, label=r"$m_p^{TEI}$")
        ax_p.plot(mi_ld, mp_alm, label=r"$m_p^{ALM}$")
        ax_p.plot(mi_ld, mp_plm, label=r"$m_p^{PLM}$")
        if self.lowspeed_reentry:
            ax_p.plot(mi_ld, mp_lsr, label=r"$m_p^{LSR}$")
        # ax_p.vlines((2445 + 2034) * 1e-3, 0., max(mp_ld), colors='k', linestyles='--')

        ax_p.tick_params(axis='both', which='major', labelsize=16)
        ax_p.set_xlabel(r"$m_i^{lander}$ (mT)", fontsize=16)
        ax_p.set_ylabel("Propellant Mass (mT)", fontsize=16)
        ax_p.legend(loc='upper right', fontsize=16)
        ax_p.grid()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
        fig.tight_layout(pad=10.)
        ax[0].set_prop_cycle(color=mycolors)
        ax[0].plot(mi_ld, mf_ld, label=r"$m_f^{LD}$", linestyle="dashed")
        ax[0].plot(mi_ld, mf_la, label=r"$m_f^{LA}$", linestyle="dashed")
        ax[0].plot(mi_ld, mf_tei, label=r"$m_f^{TEI}$")
        ax[0].plot(mi_ld, mf_alm, label=r"$m_f^{ALM}$")
        ax[0].plot(mi_ld, mf_plm, label=r"$m_f^{PLM}$")
        if self.lowspeed_reentry:
            ax[0].plot(mi_ld, mf_lsr, label=r"$m_f^{LSR}$")
            raw_final_mass = mf_lsr
            adjusted_final_mass = mf_lsr - mf_la
        else:
            raw_final_mass = mf_plm
            adjusted_final_mass = mf_plm - mf_la
        ax[0].plot(mi_ld, adjusted_final_mass, label=r"$m_f^{PLM} - m_f^{LA}$")
        # ax[0].vlines((2445 + 2034) * 1e-3, 0., max(mi_ld), colors='k', linestyles='--')
        
        fig.suptitle(plot_title, fontsize=24)
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].set_xlabel(r"$m_i^{lander}$ (mT)", fontsize=16)
        ax[0].set_ylabel("Final Mass (mT)", fontsize=16)
        ax[0].legend(loc='upper right', fontsize=16)
        ax[0].grid()

        ax[1].plot(mf_la, raw_final_mass / self.initial_mass, "Black", linestyle="dashdot", label=r"with lander")
        ax[1].plot(mf_la, adjusted_final_mass / self.initial_mass, "Black", label=r"without lander")
        # ax[1].vlines((2445 + 2034) * 1e-3, 0., max(mf_plm / self.initial_mass), colors='k', linestyles='--')

        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[1].set_xlabel(r"$m_f^{lander}$ (mT)", fontsize=16)
        ax[1].set_ylabel("Mass Fraction", fontsize=16)
        ax[1].legend(loc='upper right', fontsize=16)
        ax[1].grid()
        
        if self.lowspeed_reentry:
            string_to_append = "_low-speed"
        else:
            string_to_append = "_high-speed"
        if (self.margin > 0):
            string_to_append += "-margin.png"
        else:
            string_to_append += ".png"

        result_file=open(self.path_to_results_dir + "/gto_budget_breakdown" + string_to_append + ".csv", "w")
        result_file.write(results_table)
        result_file.close()

        minidx = np.argmin(np.abs(mf_la - self.example_lunar_lander_mass))
        example_mass_fraction = (raw_final_mass / self.initial_mass)[minidx]
        example_adjusted_mass_fraction = (adjusted_final_mass / self.initial_mass)[minidx]
        if (mf_la[-1] >= self.example_lunar_lander_mass):
            ax[1].scatter(mf_la[minidx], example_mass_fraction, s=100, c='k')
            ax[1].scatter(mf_la[minidx], example_adjusted_mass_fraction, s=100, c='k')
        print("2-STAGE Mass Fraction: {:f}, 2-STAGE Lander-Removed Mass Fraction: {:f}".format(example_mass_fraction, example_adjusted_mass_fraction))

        fig_p.savefig(self.path_to_figure_dir + "/leo_prop_mass_breakdown" + string_to_append, dpi=300)    
        fig.savefig(self.path_to_figure_dir + "/leo_mass_breakdown" + string_to_append, dpi=300)

        return 0

    def GTO_to_LLO(self):
        """

        Calculate the delta-v and the transfer times for GTO to LLO (Uphoff, 1993)

        We consider a circular parking orbit around Earth (e.g., GTO) as our initial condition, and aim to insert ourselves into a circular orbit around the Moon.
        We assume that we have access to a capsule which detaches from the spacecraft, and descends to the lunar surface.

        Parameters
        ----------
        if_margin : Boolean
            Whether or not we apply a margin to the delta-v at each stage

        Returns
        -------
        table
            Table of results

        """

        initial_altitude, terminal_altitude = self.initial_gto_perigee_altitude, self.terminal_altitude
        
        gto = ConicSections(radius=self.earth_radius, rp=initial_altitude, ra=self.initial_gto_apogee_altitude) # elliptic
        intermediate_orbit = ConicSections(radius=self.earth_radius, rp=initial_altitude, ra=self.distance / 2) # elliptic
        final_orbit = ConicSections(radius=self.earth_radius, rp=initial_altitude, ra=self.distance - self.earth_radius + self.moon_radius + terminal_altitude) # elliptic
        tei_transfer_orbit = ConicSections(radius=self.earth_radius, rp=self.initial_gto_perigee_altitude, ra=self.distance - self.earth_radius + self.moon_radius + terminal_altitude)

        # TLI
        dv_tli_1, _ = self.system.raise_apogee(rp=gto.rp, rai=gto.ra, raf=intermediate_orbit.ra)
        dv_tli_2, _ = self.system.raise_apogee(rp=gto.rp, rai=intermediate_orbit.ra, raf=final_orbit.ra)
        dv_tli = dv_tli_1 + dv_tli_2
        tof_tli = (2 * np.pi * np.sqrt(intermediate_orbit.sma**3 / self.system.mu) + np.pi * np.sqrt(final_orbit.sma**3 / self.system.mu)) * SECONDS2DAYS # one intermediate phasing orbit period + half last phasing orbit period
        # LOI
        dv_loi, _ = self.calculate_delta_v_loi(final_orbit, terminal_altitude)
        tof_loi = 0.
        # LD
        self.switch_central_body() # switch central body to be the moon for {LD, LA and TEI+EOI}
        dv_ld, _ = self.system.hohmann(ri=terminal_altitude, rf=0)
        dv_ld += np.sqrt(self.system.mu / self.system.radius)
        # LA
        dv_la, _ = self.system.hohmann(ri=0, rf=terminal_altitude)
        dv_la += np.sqrt(self.system.mu / self.system.radius)
        # dv_la += (1 / 6 * 1.5) # gravity losses, moon's gravity 1/6 of earth's, typical gravity losses for earth ~2km/s
        # TEI: 
        self.switch_central_body() # switch to earth
        dv_tei, tof_tei = self.calculate_delta_v_tei(terminal_altitude, tei_transfer_orbit)
        # ALM
        dv_alm, tof_alm =  self.system.raise_apogee(rp=self.initial_gto_perigee_altitude, rai=self.distance - self.earth_radius + self.moon_radius + terminal_altitude, raf=self.initial_gto_apogee_altitude)
        # PLM
        dv_plm, tof_plm =  self.system.lower_perigee(ra=self.initial_gto_apogee_altitude, rpi=self.initial_gto_perigee_altitude, rpf=self.earth_reentry_altitude)
        if self.lowspeed_reentry:
            # LSR
            speed_in_final_reentry_orbit = np.sqrt(2 * self.earth_mu / (self.earth_reentry_altitude + self.earth_radius) - self.earth_mu / (0.5 * (self.earth_reentry_altitude + self.initial_gto_apogee_altitude + 2 * self.earth_radius)))
            dv_lsr = speed_in_final_reentry_orbit - self.minimum_reentry_speed
        # ED = ALM + PLM
        if self.lowspeed_reentry:
            dv_ed = dv_alm + dv_plm + dv_lsr
        else:
            dv_ed = dv_alm + dv_plm
        tof_ed = tof_alm + tof_plm

        dv_tot = dv_tli + dv_loi + dv_ld + dv_la + dv_tei + dv_ed
        tof_tot = tof_tli + tof_tei + tof_ed

        if self.lowspeed_reentry:
            dv_vec = np.array([dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_lsr, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot])
        else:
            dv_vec = np.array([dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot])
        lam_vec = self.get_mass_fraction((1 + self.margin) * dv_vec * 1e3)
        final_mass_at_the_end_the_stage, propellant_mass_consumed = np.zeros_like(lam_vec), np.zeros_like(lam_vec)
        for i, lam_i in enumerate(lam_vec):
            if (i==0):
                final_mass_at_the_end_the_stage[i] = lam_i * self.initial_mass
                propellant_mass_consumed[i] = (1 - lam_i) * self.initial_mass
            else:
                final_mass_at_the_end_the_stage[i] = lam_i * final_mass_at_the_end_the_stage[i - 1]
                propellant_mass_consumed[i] = (1 - lam_i) * final_mass_at_the_end_the_stage[i - 1]
        # Overwriting the parameters for the entire mission
        final_mass_at_the_end_the_stage[-3] = self.get_mass_fraction(dv_vec[-3] * 1e3) * self.initial_mass
        propellant_mass_consumed[-3] = self.get_propellant_mass_fraction(dv_vec[-3] * 1e3) * self.initial_mass
        # Overwriting the parameters for the margin entry
        final_mass_at_the_end_the_stage[-2] = 0.
        propellant_mass_consumed[-2] = 0.
        # Overwriting the parameters for the entire mission with margin
        final_mass_at_the_end_the_stage[-1] = self.get_mass_fraction(dv_vec[-1] * 1e3) * self.initial_mass
        propellant_mass_consumed[-1] = self.get_propellant_mass_fraction(dv_vec[-1] * 1e3) * self.initial_mass

        # Calculating mass fractions
        # LD
        mi_ld = np.linspace(0., final_mass_at_the_end_the_stage[1], 1000)
        mp_ld = (1 - lam_vec[2]) * mi_ld
        mf_ld = mi_ld - mp_ld
        # LA
        mi_la = mf_ld
        mp_la = (1 - lam_vec[3]) * mi_la
        mf_la = mi_la - mp_la
        # TEI
        mi_tei = final_mass_at_the_end_the_stage[1] - mi_ld + mf_la
        mp_tei = (1 - lam_vec[4]) * mi_tei
        mf_tei = mi_tei - mp_tei
        # PLM
        mi_alm = mf_tei
        mp_alm = (1 - lam_vec[5]) * mi_alm
        mf_alm = mi_alm - mp_alm
        # ALM
        mi_plm = mf_alm
        mp_plm = (1 - lam_vec[6]) * mi_plm
        mf_plm = mi_plm - mp_plm
        # LSR
        if self.lowspeed_reentry:
            mi_lsr = mf_plm
            mp_lsr = (1 - lam_vec[7]) * mi_plm
            mf_lsr = mi_lsr - mp_lsr

        print("=========================================================================================")
        if self.lowspeed_reentry:
            phase_to_print = ['TLI', 'LOI', 'LD', 'LA', 'TEI', 'ALM', 'PLM', 'LSR', 'TOTAL', 'MARGIN', 'ADJUSTED TOTAL']
            dv_to_print = [dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_lsr, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot]
            tof_to_print = [tof_tli, tof_loi, r'$\Delta_1$', r'$\Delta_2$', tof_tei, tof_alm, tof_plm, '', tof_tot, '', tof_tot]
            mi_to_print = [self.initial_mass, final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], final_mass_at_the_end_the_stage[6], self.initial_mass, '', self.initial_mass]
            mp_to_print = [propellant_mass_consumed[0], propellant_mass_consumed[1], propellant_mass_consumed[2], propellant_mass_consumed[3], propellant_mass_consumed[4], propellant_mass_consumed[5], propellant_mass_consumed[6], propellant_mass_consumed[7], propellant_mass_consumed[-3], '', propellant_mass_consumed[-1]]
            mf_to_print = [final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], final_mass_at_the_end_the_stage[6], final_mass_at_the_end_the_stage[7], final_mass_at_the_end_the_stage[-3], '', final_mass_at_the_end_the_stage[-1]]
            if (self.margin > 0):
                print("GTO to LLO Budget For Low-Speed With Margin")
            else:
                print("GTO to LLO Budget For Low-Speed Without Margin")
        else:
            phase_to_print = ['TLI', 'LOI', 'LD', 'LA', 'TEI', 'ALM', 'PLM', 'TOTAL', 'MARGIN', 'ADJUSTED TOTAL']
            dv_to_print = [dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_alm, dv_plm, dv_tot, self.margin * dv_tot, (1 + self.margin) * dv_tot]
            tof_to_print = [tof_tli, tof_loi, r'$\Delta_1$', r'$\Delta_2$', tof_tei, tof_alm, tof_plm, tof_tot, '', tof_tot]
            mi_to_print = [self.initial_mass, final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], self.initial_mass, '', self.initial_mass]
            mp_to_print = [propellant_mass_consumed[0], propellant_mass_consumed[1], propellant_mass_consumed[2], propellant_mass_consumed[3], propellant_mass_consumed[4], propellant_mass_consumed[5], propellant_mass_consumed[6], propellant_mass_consumed[-3], '', propellant_mass_consumed[-1]]
            mf_to_print = [final_mass_at_the_end_the_stage[0], final_mass_at_the_end_the_stage[1], final_mass_at_the_end_the_stage[2], final_mass_at_the_end_the_stage[3], final_mass_at_the_end_the_stage[4], final_mass_at_the_end_the_stage[5], final_mass_at_the_end_the_stage[6], final_mass_at_the_end_the_stage[-3], '', final_mass_at_the_end_the_stage[-1]]
            if (self.margin > 0):
                print("GTO to LLO Budget For High-Speed With Margin")
            else:
                print("GTO to LLO Budget For High-Speed Without Margin")
        # print("=========================================================================================")
        results = {'Type': phase_to_print, 
                   'Delta-V (km/s)': dv_to_print, 
                   'Transfer Time (days)': tof_to_print,
                   'Mass At the Start Of Phase (mT)': mi_to_print,
                   'Propellant Mass Consumed (mT)': mp_to_print,
                   'Mass At the End Of Phase (mT)': mf_to_print}
        results_table = tabulate(results, headers='keys')
        print(results_table)
        # print("=========================================================================================")
        plot_title = "GTO"
        if self.lowspeed_reentry:
            plot_title += " low-speed"
        else:
            plot_title += " high-speed"
        if (self.margin > 0):
            plot_title += " with margin"
        else:
            plot_title += " without margin"
        fig_p, ax_p = plt.subplots(figsize=(16, 10))
        ax_p.set_prop_cycle(color=mycolors)
        ax_p.plot(mi_ld, mp_ld, label=r"$m_p^{LD}$", linestyle="dashed")
        ax_p.plot(mi_ld, mp_la, label=r"$m_p^{LA}$", linestyle="dashed")
        ax_p.plot(mi_ld, mp_tei, label=r"$m_p^{TEI}$")
        ax_p.plot(mi_ld, mp_alm, label=r"$m_p^{ALM}$")
        ax_p.plot(mi_ld, mp_plm, label=r"$m_p^{PLM}$")
        
        if self.lowspeed_reentry:
            ax_p.plot(mi_ld, mp_lsr, label=r"$m_p^{LSR}$")
        # ax_p.vlines((2445 + 2034) * 1e-3, 0., max(mp_ld), colors='k', linestyles='--')

        ax_p.set_title(plot_title, fontsize=20)
        ax_p.tick_params(axis='both', which='major', labelsize=16)
        ax_p.set_xlabel(r"$m_i^{lander}$ (mT)", fontsize=16)
        ax_p.set_ylabel("Propellant Mass (mT)", fontsize=16)
        ax_p.legend(loc='upper right', fontsize=16)
        ax_p.grid()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
        fig.tight_layout(pad=10.)
        ax[0].set_prop_cycle(color=mycolors)
        ax[0].plot(mi_ld, mf_ld, label=r"$m_f^{LD}$", linestyle="dashed")
        ax[0].plot(mi_ld, mf_la, label=r"$m_f^{LA}$", linestyle="dashed")
        ax[0].plot(mi_ld, mf_tei, label=r"$m_f^{TEI}$")
        ax[0].plot(mi_ld, mf_alm, label=r"$m_f^{ALM}$")
        ax[0].plot(mi_ld, mf_plm, label=r"$m_f^{PLM}$")
        if self.lowspeed_reentry:
            ax[0].plot(mi_ld, mf_lsr, label=r"$m_f^{LSR}$")
            raw_final_mass = mf_lsr
            adjusted_final_mass = mf_lsr - mf_la
        else:
            raw_final_mass = mf_plm
            adjusted_final_mass = mf_plm - mf_la
        ax[0].plot(mi_ld, adjusted_final_mass, label=r"$m_f^{PLM} - m_f^{LA}$")
        # ax[0].vlines((2445 + 2034) * 1e-3, 0., max(mi_ld), colors='k', linestyles='--')
        
        fig.suptitle(plot_title, fontsize=24)
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].set_xlabel(r"$m_i^{lander}$ (mT)", fontsize=16)
        ax[0].set_ylabel("Final Mass (mT)", fontsize=16)
        ax[0].legend(loc='upper right', fontsize=16)
        ax[0].grid()

        ax[1].plot(mf_la, raw_final_mass / self.initial_mass, "Black", linestyle="dashdot", label=r"with lander")
        ax[1].plot(mf_la, adjusted_final_mass / self.initial_mass, "Black", label=r"without lander")
        # ax[1].vlines((2445 + 2034) * 1e-3, 0., max(mf_plm / self.initial_mass), colors='k', linestyles='--')

        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[1].set_xlabel(r"$m_f^{lander}$ (mT)", fontsize=16)
        ax[1].set_ylabel("Mass Fraction", fontsize=16)
        ax[1].legend(loc='upper right', fontsize=16)
        ax[1].grid()
        
        if self.lowspeed_reentry:
            string_to_append = "_low-speed"
        else:
            string_to_append = "_high-speed"
        if (self.margin > 0):
            string_to_append += "-margin.png"
        else:
            string_to_append += ".png"

        result_file=open(self.path_to_results_dir + "/gto_budget_breakdown" + string_to_append + ".csv", "w")
        result_file.write(results_table)
        result_file.close()

        minidx = np.argmin(np.abs(mf_la - self.example_lunar_lander_mass))
        example_mass_fraction = (raw_final_mass / self.initial_mass)[minidx]
        example_adjusted_mass_fraction = (adjusted_final_mass / self.initial_mass)[minidx]
        if (mf_la[-1] >= self.example_lunar_lander_mass):
            ax[1].scatter(mf_la[minidx], example_mass_fraction, s=100, c='k')
            ax[1].scatter(mf_la[minidx], example_adjusted_mass_fraction, s=100, c='k')
        print("2-STAGE Mass Fraction: {:f}, 2-STAGE Lander-Removed Mass Fraction: {:f}".format(example_mass_fraction, example_adjusted_mass_fraction))

        fig_p.savefig(self.path_to_figure_dir + "/gto_prop_mass_breakdown" + string_to_append, dpi=300)    
        fig.savefig(self.path_to_figure_dir + "/gto_mass_breakdown" + string_to_append, dpi=300)

        return 0


if __name__ == '__main__':

    c1 = Mission(if_lowspeed_reentry=False, if_margin=False)
    c1.LEO_to_LLO()
    c1.GTO_to_LLO()

    c2 = Mission(if_lowspeed_reentry=False, if_margin=True)
    c2.LEO_to_LLO()
    c2.GTO_to_LLO()

    c3 = Mission(if_lowspeed_reentry=True, if_margin=False)
    c3.LEO_to_LLO()
    c3.GTO_to_LLO()

    c4 = Mission(if_lowspeed_reentry=True, if_margin=True)
    c4.LEO_to_LLO()
    c4.GTO_to_LLO()