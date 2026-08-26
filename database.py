import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class DiagnosticRecord(Base):
    __tablename__ = "diagnostic_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    patient_name = Column(String(128), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(16), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    spatial_grounding = Column(String(128), nullable=False)
    clinical_report = Column(Text, nullable=False)
    # Stored as Base64 strings for single-file self-contained DB records
    original_image_b64 = Column(Text, nullable=False)
    overlay_image_b64 = Column(Text, nullable=False)

# Local SQLite file database
DATABASE_URL = "sqlite:///medical_diagnostics.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def save_record(patient_id, patient_name, age, gender, spatial_grounding, clinical_report, orig_b64, overlay_b64):
    session = SessionLocal()
    try:
        record = DiagnosticRecord(
            patient_id=patient_id,
            patient_name=patient_name,
            age=age,
            gender=gender,
            spatial_grounding=spatial_grounding,
            clinical_report=clinical_report,
            original_image_b64=orig_b64,
            overlay_image_b64=overlay_b64
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    finally:
        session.close()

def get_all_records():
    session = SessionLocal()
    try:
        return session.query(DiagnosticRecord).order_by(DiagnosticRecord.created_at.desc()).all()
    finally:
        session.close()

def get_record_by_id(record_id):
    session = SessionLocal()
    try:
        return session.query(DiagnosticRecord).filter(DiagnosticRecord.id == record_id).first()
    finally:
        session.close()