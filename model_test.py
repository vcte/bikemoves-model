from model import *

# endpoints of east washington street segment
e_wash_st_endpts = ((40.105667, -88.200227), (40.105651, -88.195678))

### tests for the helper function pt_is_between ###
def test_pt_is_between_returns_true_if_pt_in_street_segment():
    """pt_is_between should return true if pt is obviously inside of street"""
    test_pt = (40.105684, -88.198296)
    assert pt_is_between(*e_wash_st_endpts, test_pt), \
           "test pt is in between washington street endpoints"

def test_pt_is_between_returns_true_if_pt_equals_endpoint():
    """pt_is_between should return true if pt is identical
       with an endpoint of the street"""
    test_pt = (40.105667, -88.200227)
    assert pt_is_between(*e_wash_st_endpts, test_pt), \
           "test pt is in between washington street endpoints"

def test_pt_is_between_returns_true_if_pt_slightly_off():
    """pt_is_between should return true if pt is slightly off the street"""
    test_pt = (40.105897, -88.197888)
    assert pt_is_between(*e_wash_st_endpts, test_pt, tol = 10), \
           "test pt is slightly outside of washington street"

def test_pt_is_between_returns_false_if_pt_outside_and_collinear():
    """pt_is_between should return false if pt is collinear w/ segment,
       but not in between the endpoints"""
    test_pt = (40.105651, -88.201279)
    assert not pt_is_between(*e_wash_st_endpts, test_pt), \
           "test pt is outside, but collinear w/ washington street endpoints"

def test_pt_is_between_returns_false_if_pt_on_different_street():
    """pt_is_between should return false if pt is on different street"""
    test_pt = (40.107719, -88.200227)
    assert not pt_is_between(*e_wash_st_endpts, test_pt, tol = 10), \
           "test pt is not on washington street"

def test_pt_is_between_returns_false_if_tolerance_lower():
    """pt_is_between should return false if pt is slightly off the street,
       and the tolerance is extremely low"""
    test_pt = (40.105897, -88.197888)
    assert not pt_is_between(*e_wash_st_endpts, test_pt, tol = 1), \
           "test pt is slightly outside of washington street"

def test_pt_is_between():
    test_pt_is_between_returns_true_if_pt_in_street_segment()
    test_pt_is_between_returns_true_if_pt_equals_endpoint()
    test_pt_is_between_returns_true_if_pt_slightly_off()
    test_pt_is_between_returns_false_if_pt_outside_and_collinear()
    test_pt_is_between_returns_false_if_pt_on_different_street()
    test_pt_is_between_returns_false_if_tolerance_lower()

### main test function ###
def test_all():
    test_pt_is_between()
    print("All test cases passed")

if __name__ == "__main__":
    test_all()
    
