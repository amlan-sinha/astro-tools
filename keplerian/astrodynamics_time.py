from numbers import Real
from numpy import ndarray, array


def julian_date_to_modified_julian_date(julian_date: Real) -> Real:
    """ JD -> MJD """
    assert isinstance(julian_date, Real)
    return julian_date - 2400000.5


def julian_date_to_modified_J2000_second(julian_date: Real) -> Real:
    """ JD -> MJD -> shifted to J2000 -> converted to seconds """
    assert isinstance(julian_date, Real)
    return (julian_date_to_modified_julian_date(julian_date) - 51544.5) * 86400.


def modified_julian_date_to_julian_date(modified_julian_date: Real) -> Real:
    """ MJD -> JD """
    assert isinstance(modified_julian_date, Real)
    return modified_julian_date + 2400000.5


def modified_julian_date_to_relative_MJS(modified_julian_date: Real, shift: float = 51544.5) -> Real:
    return (modified_julian_date - shift) * 86400.


def modified_J2000_second_to_julian_date(mjs: Real, shift: float = 51544.5) -> Real:
    modified_julian_date = (mjs / 86400.) + shift
    return modified_julian_date_to_julian_date(modified_julian_date)


def gregorian_date_to_julian_date(gregorian_date: (list, None)=None,
                                  year: (int, None)=None,
                                  month: (int, None)=None,
                                  day: (int, None)=None,
                                  hour: (int, None)=None,
                                  minute: (int, None)=None,
                                  second: (int, None)=None) -> Real:

    if gregorian_date is not None:
        assert isinstance(gregorian_date, list)
        assert len(gregorian_date) == 6
        year = gregorian_date[0]
        month = gregorian_date[1]
        day = gregorian_date[2]
        hour = gregorian_date[3]
        minute = gregorian_date[4]
        second = gregorian_date[5]

    assert isinstance(year, int)
    assert isinstance(month, int)
    assert 1 <= month <= 12
    assert isinstance(day, int)
    assert 1 <= day <= 31
    assert isinstance(hour, int)
    assert 0 <= hour <= 23
    assert isinstance(minute, int)
    assert 0 <= minute <= 59
    assert isinstance(second, int)
    assert 0 <= second <= 59

    return 367. * year - int(7. * (year + int((month + 9.) / 12.)) / 4.) + int(275. * month / 9.) + day + 1721013.5 \
           + (((minute + second / 60.) / 60.) + hour) / 24.


def julian_date_to_gregorian_date(julian_date: Real) -> ndarray:

    shift = 2415019.5

    t1900 = (julian_date - shift) / 365.25

    year = 1900 + int(t1900)

    leap_years = int((year - 1900 - 1) * .25)

    days = (julian_date - shift) - ((year - 1900) * 365. + leap_years)

    if days < 1.:
        year = year - 1
        leap_years = int((year - 1900 - 1) * .25)
        days = (julian_date - shift) - ((year - 1900) * 365. + leap_years)

    month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if year % 4 == 0:
        month_length[1] += 1

    day_of_year = int(days)

    month = 0
    summation = 0
    while summation < day_of_year:
        summation += month_length[month]
        month += 1

    day = day_of_year - summation + month_length[month - 1]

    tau = (days - day_of_year) * 24.
    hour = int(tau)
    minute = int((tau - hour) * 60.)
    second = (tau - hour - (minute / 60)) * 3600.

    return array([year, month, day, hour, minute, second])