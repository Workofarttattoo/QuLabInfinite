"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NUCLEAR PHYSICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi, physical_constants

@dataclass
class NuclearPhysicsConstants:
    k: float = field(default=k)
    Avogadro: float = field(default=Avogadro)
    g: float = field(default=g)
    c: float = field(default=c)
    h: float = field(default=h)
    e: float = field(default=e)
    pi: float = field(default=pi)

def parse_constant(constant_name):
    value, unit, uncertainty = physical_constants[constant_name]
    return value

class Nucleus:
    def __init__(self, z: int, a: int, r0: float = 1.25e-15) -> None:
        self.z = z
        self.a = a
        self.r0 = r0
    
    @property
    def radius(self) -> np.ndarray:
        return (self.r0 * self.a**(1/3))
    
    @property
    def volume(self) -> np.ndarray:
        return (4/3) * pi * self.radius**3
    
    @property
    def mass_number_density(self) -> np.ndarray:
        return 1 / self.volume

@dataclass
class NucleusCollection:
    nuclei: list[Nucleus] = field(default_factory=list)

    def add_nucleus(self, nucleus: Nucleus):
        self.nuclei.append(nucleus)
    
    @property
    def total_volume(self) -> np.ndarray:
        return sum([nucleus.volume for nucleus in self.nuclei], 0.0).astype(np.float64)

@dataclass
class NuclearReaction:
    reactants: NucleusCollection = field(default_factory=NucleusCollection)
    products: NucleusCollection = field(default_factory=NucleusCollection)
    
    def add_reaction(self, reactant_z: int, reactant_a: int, product_z: int, product_a: int):
        reactant_nucleus = Nucleus(reactant_z, reactant_a)
        self.reactants.add_nucleus(reactant_nucleus)

        product_nucleus = Nucleus(product_z, product_a)
        self.products.add_nucleus(product_nucleus)

def run_demo():
    constants = NuclearPhysicsConstants()
    
    nucleus_collection_1 = NucleusCollection([Nucleus(z=26, a=56), Nucleus(z=30, a=60)])
    nucleus_collection_2 = NucleusCollection([Nucleus(z=28, a=58), Nucleus(z=24, a=56)])

    nuclear_reaction = NuclearReaction()
    
    nuclear_reaction.add_reaction(reactant_z=26, reactant_a=56, product_z=30, product_a=60)
    nuclear_reaction.add_reaction(reactant_z=28, reactant_a=58, product_z=24, product_a=56)

    print(f"Reaction: {nuclear_reaction.reactants.nuclei} -> {nuclear_reaction.products.nuclei}")
    
run_demo()