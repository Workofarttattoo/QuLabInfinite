import datetime
from .schemas import RecordChem, RecordMaterial
from sqlalchemy import JSON, DateTime, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class RecordChemDB(Base):
    __tablename__ = 'record_chem'

    id = Column(Integer, primary_key=True)
    substance = Column(String)
    phase = Column(String, nullable=True)
    pressure_pa = Column(Float)
    temperature_k = Column(Float)
    volume_m3_per_mol = Column(Float, nullable=True)
    enthalpy_j_per_mol = Column(Float, nullable=True)
    entropy_j_per_mol_k = Column(Float, nullable=True)
    composition = Column(JSON, nullable=True)
    experiment_id = Column(String, nullable=True)
    tags = Column(JSON)
    provenance = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class RecordMaterialDB(Base):
    __tablename__ = 'record_material'

    id = Column(Integer, primary_key=True)
    substance = Column(String)
    material_id = Column(String, unique=True)
    phase = Column(String, default='solid')
    structure = Column(JSON)
    formation_energy_per_atom_ev = Column(Float)
    band_gap_ev = Column(Float, nullable=True)
    density_g_cm3 = Column(Float, nullable=True)
    volume_a3_per_atom = Column(Float, nullable=True)
    formation_energy_per_atom_j = Column(Float, nullable=True)
    band_gap_j = Column(Float, nullable=True)
    density_kg_m3 = Column(Float, nullable=True)
    volume_m3_per_atom = Column(Float, nullable=True)
    tags = Column(JSON)
    provenance = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def setup_database(db_path: str):
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

