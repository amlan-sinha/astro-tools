import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

import sys

DAYS2SECONDS=86400
SECONDS2DAYS=1./86400

if sys.platform.startswith("darwin"):
    sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
    sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')
    sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/extra/mission_scripts/my_support')
    sys.path.append('./')
    sys.path.append('../')

import pydylan

from keplerian.tools import Kepler, ConicSections
from keplerian.astrodynamics_time import gregorian_date_to_julian_date, julian_date_to_modified_J2000_second

class Mission():

    def __init__(self):

        sun, earth, moon = pydylan.Body("Sun"), pydylan.Body("Earth"), pydylan.Body("Moon")

        self.system, self.other = Kepler(earth), Kepler(moon)
        self.earth_radius, self.moon_radius = earth.radius, moon.radius
        self.earth_mu, self.moon_mu = earth.mu, moon.mu

        self.eom = pydylan.eom.Ephemeris_nBP(earth)
        self.eom.add_secondary_body(moon)
        self.eom.add_secondary_body(sun)

        self.initial_altitude, self.terminal_altitude = 500, 100
        initial_perigee, initial_apogee = self.initial_altitude + self.earth_radius, 35975 + self.earth_radius
        initial_sma, initial_eccentricity = 0.5 * (initial_perigee + initial_apogee), (initial_apogee - initial_perigee) / (initial_apogee + initial_perigee)
        terminal_perigee, terminal_apogee = self.terminal_altitude + self.moon_radius, self.terminal_altitude + self.moon_radius
        terminal_sma, terminal_eccentricity = 0.5 * (terminal_perigee + terminal_apogee), (terminal_apogee - terminal_perigee) / (terminal_apogee + terminal_perigee)

        self.epoch = julian_date_to_modified_J2000_second(gregorian_date_to_julian_date(year=2030,
                                                                                        month=1,
                                                                                        day=1,
                                                                                        hour=0,
                                                                                        minute=0,
                                                                                        second=0))

        earth_eom = pydylan.eom.S2BP(earth)
        initial_state = earth_eom.coe2rv(initial_sma,
                                         initial_eccentricity,
                                         -5.14,
                                         0.,
                                         0.,
                                         0.)
        self.initial_state = np.concatenate((initial_state[0], initial_state[1]))

        moon_eom = pydylan.eom.S2BP(moon)
        terminal_state = moon_eom.coe2rv(terminal_sma,
                                         terminal_eccentricity,
                                         0.,
                                         0.,
                                         0.,
                                         0.)
        self.terminal_state = np.concatenate((terminal_state[0], terminal_state[1]))
        self.terminal_state += moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(self.epoch)

        self.distance = np.linalg.norm(moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(self.epoch)[:3])
        self.velocity = np.linalg.norm(moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(self.epoch)[3:])
        self.isp = 330
        self.g0 = 9.81

    def switch_central_body(self):
        tmp = self.system
        self.system = self.other
        self.other = tmp

    def get_mass_fraction(self, dv):

        return 1 - np.exp(- dv / self.isp / self.g0)

    def get_delta_v(self, control, number_of_segments) -> float:
        delta_v = 0.
        for i in range(number_of_segments):
            starting_index = i * 3
            ending_index = (i + 1) * 3
            delta_v += abs(np.linalg.norm(control[starting_index:ending_index]))

        return delta_v

    def output_control_to_screen(self, control):
        print('\nOptimal Control Input:')
        for entry in control:
            print('{},'.format(entry))

    def calculate_delta_v_loi(self, initial_orbit, terminal_altitude):
        ''' Calculate the velocity in the MCI '''

        terminal_altitude += self.moon_radius

        v_moon_rel_earth = self.velocity  # mean orbital velocity of the moon in ECI
        v_sc_rel_earth = np.sqrt(2 * self.system.mu / initial_orbit.ra - self.system.mu / initial_orbit.sma)  # orbital velocity of the spacecraft in ECI
        v_sc_inf = np.sqrt(v_moon_rel_earth**2 + v_sc_rel_earth**2 - 2 * v_moon_rel_earth * v_sc_rel_earth)  # excess velocity assuming inclination and fpa to be zero
        v_sc_rel_moon = np.sqrt(v_sc_inf**2 + 2 * self.moon_mu / terminal_altitude)  # spacecraft orbital velocity in LCI
        
        return np.abs(v_sc_rel_moon - np.sqrt(self.moon_mu / terminal_altitude))

    def calculate_delta_v_tei(self, initial_altitude, terminal_altitude):
        ''' Calculate the velocity in the MCI '''

        initial_altitude += self.earth_radius
        terminal_altitude += self.moon_radius

        transfer_sma = 0.5 * (initial_altitude + self.distance + terminal_altitude)

        v_moon_rel_earth = self.velocity  # np.sqrt(self.earth_mu / self.distance) # mean orbital velocity of the moon in ECI
        v_sc_rel_earth = np.sqrt(2 * self.earth_mu / self.distance - self.earth_mu / transfer_sma) # orbital velocity of the spacecraft in ECI
        v_sc_inf = np.abs(v_moon_rel_earth - v_sc_rel_earth) # excess velocity assuming inclination and fpa to be zero
        v_sc_rel_moon = np.sqrt(v_sc_inf**2 + 2 * self.moon_mu / terminal_altitude) # spacecraft orbital velocity in LCI

        return abs(v_sc_rel_moon - np.sqrt(self.moon_mu / terminal_altitude)), np.pi * np.sqrt(transfer_sma**3 / self.earth_mu) * SECONDS2DAYS

    def concept_1(self):
        '''
        Analyzing Hohmann (and Bielliptic) Transfers

        We consider a circular parking orbit around Earth (e.g., LEO) as our initial condition, and aim to insert ourselves into a circular orbit around the Moon.
        We assume that we have access to a capsule which detaches from the spacecraft, and descends to the lunar surface.

        '''
        initial_altitude, terminal_altitude = self.initial_altitude, self.terminal_altitude

        initial_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=initial_altitude) # circular
        intermediate_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=self.distance / 2) # elliptic (to be used in bi-elliptic transfer), perigee same as initial altitude, apogee arbitrarily chosen [does not affect Delta V]
        final_orbit = ConicSections(radius=self.system.radius, rp=self.distance + self.moon_radius + terminal_altitude, ra=self.distance + self.moon_radius + terminal_altitude) # circular

        # TLI
        dv_tli_hoh, tof_tli_hoh = self.system.hohmann(ri=initial_orbit.rp, rf=final_orbit.ra)
        dv_tli_bie, tof_tli_bie = self.system.bielliptic(ri=initial_orbit.rp, rb=intermediate_orbit.ra, rf=final_orbit.ra)
        dv_tli = min(dv_tli_hoh, dv_tli_bie)
        tof_tli = tof_tli_hoh if dv_tli == dv_tli_hoh else tof_tli_bie
        if ((dv_tli_bie < dv_tli_hoh) and (tof_tli_bie > 2 * tof_tli_hoh)):
            print("Going forward with Hohmann Transfer", flush=True)
            dv_tli, tof_tli = dv_tli_hoh, tof_tli_hoh
        # LOI
        dv_loi = self.calculate_delta_v_loi(final_orbit, self.terminal_altitude)
        # LD
        self.switch_central_body() # switch central body to be the moon for {LD, LA and EOI}
        dv_ld, _ = self.system.hohmann(ri=terminal_altitude, rf=0) # from the LLO to the surface of the moon
        dv_ld += np.sqrt(self.system.mu / self.system.radius) # excess velocity at the surface of the moon
        # LA
        dv_la, _ = self.system.hohmann(ri=0, rf=terminal_altitude) # from the surface to the LLO
        dv_la += np.sqrt(self.system.mu / self.system.radius) # excess velocity at the surface of the moon
        dv_la += (1 / 6 * 1.5) # gravity loss: typical gravity losses for earth ~2km/s, moon's gravity 1/6 of earth's
        # TEI
        self.switch_central_body() # switch to earth
        dv_tei, tof_tei = self.calculate_delta_v_tei(initial_altitude, terminal_altitude)
        # ED
        dv_ed, _ =  self.system.raise_apogee(rp=initial_altitude, rai=self.distance + self.moon_radius + terminal_altitude, raf=0) 
        # dv_ed += np.sqrt(self.system.mu / self.system.radius) # excess velocity at the surface of the earth

        dv_tot = dv_tli + dv_loi + dv_ld + dv_la + dv_tei + dv_ed
        tof_tot = tof_tli + tof_tei

        print("========================================================")
        results = {'Type': ['TLI', 'LOI', 'LD', 'LA', 'TEI', 'ED', 'TOTAL', 'MARGIN (20%)', 'ADJUSTED TOTAL'], 'Delta-V (km/s)': [dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_ed, dv_tot, 0.2 * dv_tot, 1.2 * dv_tot], 'Transfer Time (days)': [tof_tli, 0., 'n/a', 'n/a', tof_tei, 'n/a', tof_tot, '', '']}
        print(tabulate(results, headers='keys'))
        print("========================================================")

        print("========================================================")
        results = {'Type': ['LD', 'LA', 'LD w/ margin', 'LA w/ margin'], 'Delta-V (km/s)': [dv_ld, dv_la, 1.2 * dv_ld, 1.2 * dv_la], 'Propellant Mass Margin': [self.get_mass_fraction(dv_ld * 1e3), self.get_mass_fraction(dv_la * 1e3), self.get_mass_fraction(1.2 * dv_ld * 1e3), self.get_mass_fraction(1.2 * dv_la * 1e3)]}
        print(tabulate(results, headers='keys'))
        print("========================================================")

        print("MASS FRACTION (w/o margin): {:.3f}".format(self.get_mass_fraction(dv_tot * 1e3)))
        print("MASS FRACTION (w/ margin): {:.3f}".format(self.get_mass_fraction(1.2 * dv_tot * 1e3)))

        return 0

    def concept_2(self):
        '''
        Uphoff (1993)

        We consider a GTO around Earth (200 x 35975) as our initial condition, and aim to insert ourselves into a circular orbit around the Moon.
        We assume that we have access to a capsule which detaches from the spacecraft, and descends to the lunar surface.
        For the return leg, we consider the same elliptic orbit as the last earth phasing and wait long enough to arrive at the perigee of this phasing orbit.
        Then, we perform a burn to descend to the earth surface.

        '''

        initial_altitude, terminal_altitude = self.initial_altitude, self.terminal_altitude
        
        initial_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=35975) # elliptic
        intermediate_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=self.distance / 2) # elliptic
        final_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=self.distance + self.moon_radius + terminal_altitude) # elliptic

        # TLI
        dv_tli_1, _ = self.system.raise_apogee(rp=initial_orbit.rp, rai=initial_orbit.ra, raf=intermediate_orbit.ra)
        dv_tli_2, _ = self.system.raise_apogee(rp=initial_orbit.rp, rai=intermediate_orbit.ra, raf=final_orbit.ra)
        dv_tli = dv_tli_1 + dv_tli_2
        tof_tli = (2 * np.pi * np.sqrt(intermediate_orbit.sma**3 / self.system.mu) + np.pi * np.sqrt(final_orbit.sma**3 / self.system.mu)) * SECONDS2DAYS # one intermediate phasing orbit period + half last phasing orbit period
        # LOI
        dv_loi = self.calculate_delta_v_loi(final_orbit, self.terminal_altitude)
        # LD
        self.switch_central_body() # switch central body to be the moon for {LD, LA and TEI+EOI}
        dv_ld, _ = self.system.hohmann(ri=terminal_altitude, rf=0)
        dv_ld += np.sqrt(self.system.mu / self.system.radius)
        # LA
        dv_la, _ = self.system.hohmann(ri=0, rf=terminal_altitude)
        dv_la += np.sqrt(self.system.mu / self.system.radius)
        dv_la += (1 / 6 * 1.5) # gravity losses, moon's gravity 1/6 of earth's, typical gravity losses for earth ~2km/s
        # TEI: 
        self.switch_central_body() # switch to earth
        dv_tei, tof_tei = self.calculate_delta_v_tei(initial_altitude, terminal_altitude)
        # ED
        dv_ed, _ =  self.system.raise_apogee(rp=initial_altitude, rai=self.distance + self.moon_radius + terminal_altitude, raf=0)
        # dv_ed += np.sqrt(self.system.mu / self.system.radius) # excess velocity at the surface of the earth

        dv_tot = dv_tli + dv_loi + dv_ld + dv_la + dv_tei + dv_ed
        tof_tot = tof_tli + tof_tei

        print("========================================================")
        results = {'Type': ['TLI', 'LOI', 'LD', 'LA', 'TEI', 'ED', 'TOTAL', 'MARGIN (20%)', 'ADJUSTED TOTAL'], 'Delta-V (km/s)': [dv_tli, dv_loi, dv_ld, dv_la, dv_tei, dv_ed, dv_tot, 0.2 * dv_tot, 1.2 * dv_tot], 'Transfer Time (days)': [tof_tli, 0., 'n/a', 'n/a', tof_tei, 'n/a', tof_tot, '', '']}
        print(tabulate(results, headers='keys'))
        print("========================================================")

        print("========================================================")
        results = {'Type': ['LD', 'LA', 'LD w/ margin', 'LA w/ margin'], 'Delta-V (km/s)': [dv_ld, dv_la, 1.2 * dv_ld, 1.2 * dv_la], 'Propellant Mass Margin': [self.get_mass_fraction(dv_ld * 1e3), self.get_mass_fraction(dv_la * 1e3), self.get_mass_fraction(1.2 * dv_ld * 1e3), self.get_mass_fraction(1.2 * dv_la * 1e3)]}
        print(tabulate(results, headers='keys'))
        print("========================================================")

        print("MASS FRACTION (w/o margin): {:.3f}".format(self.get_mass_fraction(dv_tot * 1e3)))
        print("MASS FRACTION (w/ margin): {:.3f}".format(self.get_mass_fraction(1.2 * dv_tot * 1e3)))

        return 0

    def concept_3(self):
        '''
        Direct Transfer

        We consider the earth surface as our initial condition, and aim to directly land on the moon.
        While this may not a feasible trajectory to fly, we intend to use this to obtain an upper bound on the fuel expenditure.

        '''

        initial_altitude, terminal_altitude = self.initial_altitude, 0.
        
        initial_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=35975) # elliptic
        intermediate_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=self.distance / 2) # elliptic
        final_orbit = ConicSections(radius=self.system.radius, rp=initial_altitude, ra=self.distance + self.moon_radius + terminal_altitude) # elliptic

        # TLI:
        dv_tli_1, _ = self.system.raise_apogee(rp=initial_orbit.rp, rai=initial_orbit.ra, raf=intermediate_orbit.ra)
        dv_tli_2, _ = self.system.raise_apogee(rp=initial_orbit.rp, rai=intermediate_orbit.ra, raf=final_orbit.ra)
        dv_tli = dv_tli_1 + dv_tli_2
        tof_tli = (2 * np.pi * np.sqrt(intermediate_orbit.sma**3 / self.system.mu) + np.pi * np.sqrt(final_orbit.sma**3 / self.system.mu)) * SECONDS2DAYS # one intermediate phasing orbit period + half last phasing orbit period
        # LOI
        dv_loi = self.calculate_delta_v_loi(final_orbit, self.terminal_altitude)
        dv_loi += np.sqrt(self.moon_mu / self.moon_radius)
        # TEI
        dv_tei, tof_tei = self.calculate_delta_v_tei(initial_altitude, terminal_altitude)
        dv_tei += np.sqrt(self.moon_mu / self.moon_radius)

        dv_tot = dv_tli + dv_loi + dv_tei
        tof_tot = tof_tli + tof_tei

        print("========================================================")
        results = {'Type': ['TLI', 'LOI', 'TEI', 'TOTAL', 'MARGIN (20%)', 'ADJUSTED TOTAL'], 'Delta-V (km/s)': [dv_tli, dv_loi, dv_tei, dv_tot, 0.2 * dv_tot, 1.2 * dv_tot], 'Transfer Time (days)': [tof_tli, 0., tof_tei, tof_tot, '', '']}
        print(tabulate(results, headers='keys'))

        print("MASS FRACTION (w/o margin): {:.3f}".format(self.get_mass_fraction(dv_tot * 1e3)))
        print("MASS FRACTION (w/ margin): {:.3f}".format(self.get_mass_fraction(1.2 * dv_tot * 1e3)))

        return 0

    def concept_4(self):

        pydylan.set_logging_severity(pydylan.enum.LogLevel.error)

        snopt_options = pydylan.SNOPT_options_structure()
        snopt_options.solver_mode = pydylan.enum.solver_mode_type.optimal
        snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.finite_differencing
        # snopt_options.enable_SNOPT_auto_scale = True
        snopt_options.quiet_SNOPT = False
        snopt_options.time_limit = 5 * 60

        mbh_options = pydylan.MBH_options_structure()
        mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
        mbh_options.quiet_MBH = False
        mbh_options.time_limit = 3 * 60 * 60
        mbh_options.number_of_solutions_to_save = 10

        phase_options = pydylan.phase_options_structure()
        phase_options.earliest_initial_date_in_MJS = self.epoch
        phase_options.number_of_segments = 4
        phase_options.minimum_initial_coast_time = 0.
        phase_options.maximum_initial_coast_time = DAYS2SECONDS
        phase_options.minimum_shooting_time = 0.
        phase_options.maximum_shooting_time = 30 * DAYS2SECONDS
        phase_options.minimum_final_coast_time = 0.
        phase_options.maximum_final_coast_time = DAYS2SECONDS
        phase_options.match_point_position_constraint_tolerance = 1E+3
        phase_options.match_point_velocity_constraint_tolerance = 1E-1
        phase_options.control_coordinate_transcription = pydylan.enum.cartesian # pydylan.enum.spherical # pydylan.enum.cartesian
        phase_options.transcription = pydylan.enum.transcription_type.ForwardBackwardShooting
        phase_options.optimal_control_methodology = pydylan.enum.optimal_control_methodology_type.Direct

        # left_boundary_condition = pydylan.AdvectingBoundaryCondition(initial_state, epoch, eom)
        # right_boundary_condition = pydylan.AdvectingBoundaryCondition(terminal_state, epoch, eom)
        left_boundary_condition = pydylan.FixedBoundaryCondition(self.initial_state)
        right_boundary_condition = pydylan.FixedBoundaryCondition(self.terminal_state)

        thruster = pydylan.ThrustParameters(fuel_mass=0., dry_mass=0., Isp=330., thrust=0., duty_cycle=1., is_low_thrust=False, maximum_impulse=1.)

        mission = pydylan.Mission(self.eom, left_boundary_condition, right_boundary_condition, pydylan.enum.mbh)
        mission.set_random_number_generator_seed(0)
        mission.set_thruster_parameters(thruster)
        mission.add_phase_options(phase_options)
        mission.optimize(snopt_options, mbh_options)

        np.save("./data/c4_feasible.npy", mission.get_all_feasible_control_solutions())

        # assert mission.is_best_solution_feasible()

        total_transfer_time = mission.get_control_state()[0] + mission.get_control_state()[1] + mission.get_control_state()[2]

        rk54 = pydylan.integrators.RK54()
        rk54.set_eom(self.eom)
        rk54.set_time(self.epoch, self.epoch + total_transfer_time)
        rk54.evaluate(self.initial_state)
        earth_propagated_states = rk54.get_states()

        rk54.evaluate(self.terminal_state)
        moon_propagated_states = rk54.get_states()
        
        results = mission.evaluate_and_return_solution(mission.get_control_state(), pydylan.enum.transcription_type.ForwardBackwardShooting, 1E-1)
        time, states = results.time, results.states

        print("========================================================")
        print('Initial coast time: {} (days)'.format(mission.get_control_state()[1] * SECONDS2DAYS))
        print('Transfer time: {} (days)'.format(mission.get_control_state()[0] * SECONDS2DAYS))
        print('Final coast time: {} (days)'.format(mission.get_control_state()[2] * SECONDS2DAYS))
        print('Total mission time: {} (days)'.format((total_transfer_time) * SECONDS2DAYS))
        print('Total Delta-v: {} (km/s)'.format(self.get_delta_v(mission.get_control_state()[3:], phase_options.number_of_segments)))
        print('Objective: {}'.format(results.objective_value))
        print('\nConstraint Violation:')
        print(results.constraint_violations)
        self.output_control_to_screen(mission.get_control_state())
        print("========================================================")

        fig_xy, ax_xy = plt.subplots()
        fig_xz, ax_xz = plt.subplots()

        # ax_xy.plot(earth_propagated_states[:, 0], earth_propagated_states[:, 1], color='DodgerBlue')
        ax_xy.scatter(0, 0, color='DodgerBlue')
        ax_xy.plot(moon_propagated_states[:, 0], moon_propagated_states[:, 1], color='DarkGrey')
        ax_xy.plot(states[:, 0], states[:, 1], color='Chartreuse')
        

        # ax_xz.plot(earth_propagated_states[:, 0], earth_propagated_states[:, 2], color='DodgerBlue')
        ax_xz.scatter(0, 0, color='DodgerBlue')
        ax_xz.plot(moon_propagated_states[:, 0], moon_propagated_states[:, 2], color='DarkGrey')
        ax_xz.plot(states[:, 0], states[:, 2], color='Chartreuse')

        fig_xy.savefig("./data/c3_optimal_xy.png", dpi=300)
        fig_xz.savefig("./data/c3_optimal_xz.png", dpi=300)

        plt.show()

        return 0

    def launch_vehicle_insertion(self, target_rp, target_ra):
        '''
        LV Insertion
        '''
        earth = Kepler(pydylan.Body("Earth"))
        if (target_rp == target_ra):
            dv, _ = earth.hohmann(ri=0., rf=target_rp)
        else:
            dv1, _ = earth.raise_apogee(rp=0., rai=0., raf=target_rp)
            dv2, _ = earth.raise_apogee(rp=0., rai=target_rp, raf=target_ra)
            dv = dv1 + dv2

        dv += np.sqrt(self.earth_mu / self.earth_radius)
        dv += 1.5 # gravity and atmospheric losses

        return dv 

if __name__ == '__main__':

    kernels = pydylan.spice.load_spice()

    c = Mission()
    c.concept_1()
    c.concept_2()
    c.concept_3()

    # # Writing out the feasible solutions
    # dv, tof for various feasible
    # a = np.load("./data/c3_feasible.npy")
    # dv = []
    # tof = []
    # for ai in a:
    #     dv.append(c.get_delta_v(ai[3:], 4))
    #     tof.append((ai[0] + ai[1] + ai[2]) * SECONDS2DAYS)

    # print("dv: ", dv)
    # print("tof: ", tof)

    # # Analyzing the LEO vs GTO LV Insertion
    # dv_vec_1, tof_vec_1 = [], []
    # dv_vec_2a, tof_vec_2a = [], []
    # dv_vec_2b, tof_vec_2b = [], []
    # dv_vec_2c, tof_vec_2c = [], []
    # rpvec, ravec = np.linspace(0, 2500), 35975
    # for i, rp in enumerate(rpvec):
    #     dv_hoh = c.launch_vehicle_insertion(target_rp=rp, target_ra=rp)
    #     dv_ell_1 = c.launch_vehicle_insertion(target_rp=rp, target_ra=0.5 * ravec)
    #     dv_ell_2 = c.launch_vehicle_insertion(target_rp=rp, target_ra=ravec)
    #     dv_ell_3 = c.launch_vehicle_insertion(target_rp=rp, target_ra=1.5 * ravec)
    #     dv_vec_1.append(dv_hoh)
    #     dv_vec_2a.append(dv_ell_1)
    #     dv_vec_2b.append(dv_ell_2)
    #     dv_vec_2c.append(dv_ell_3)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax_twin = ax.twinx()
    # ax.plot(rpvec, np.asarray(dv_vec_1), color='black', linewidth=2, label='LEO')
    # ax_twin.plot(rpvec, np.asarray(dv_vec_2a), color='LightSkyBlue', linewidth=2, label=r'GTO ($r_a$ = {:f} km)'.format(0.5 * ravec))
    # ax_twin.plot(rpvec, np.asarray(dv_vec_2b), color='DodgerBlue', linewidth=2, label=r'GTO ($r_a$ = {:f} km)'.format(ravec))
    # ax_twin.plot(rpvec, np.asarray(dv_vec_2c), color='DarkTurquoise', linewidth=2, label=r'GTO ($r_a$ = {:f} km)'.format(1.5 * ravec))
    # ax.set_xlabel(r"$r_P$ (km)", fontsize=16)
    # ax.set_ylabel(r"$\Delta V_{LEO}$ (km/s)", fontsize=16)
    # ax_twin.set_ylabel(r"$\Delta V_{GTO}$ (km/s)", fontsize=16, color='DodgerBlue')
    # ax_twin.tick_params(axis='y', labelcolor='DodgerBlue')
    # ax.legend(loc='lower right')
    # ax_twin.legend(loc='upper right')
    # ax.set_title(r"Launch Vehicle Insertion: LEO vs GTO", fontsize=18)

    # fig.savefig("./data/leo_vs_gto.png", dpi=300)

    # plt.show()

    pydylan.spice.unload_spice(kernels)