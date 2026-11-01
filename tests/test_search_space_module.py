"""The production search space is every instrument's (klip_tpe.instruments.search_space).

It lived in instruments/near.py, so an ALES change to its own space edited NEAR's module.  It
moved on 2026-11-01, and near.make_space, near.pa_span_max and near.production_defaults still
name the same objects, so every script that builds a space through near keeps working.
"""
from klip_tpe.instruments import near, nomic, search_space


def test_the_space_builder_is_shared_and_near_keeps_its_names():
    assert near.make_space is search_space.make_space
    assert near.pa_span_max is search_space.pa_span_max
    assert near.production_defaults is search_space.production_defaults
    assert nomic.production_defaults == search_space.production_defaults
