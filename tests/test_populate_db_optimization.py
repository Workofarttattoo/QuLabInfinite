
import unittest
from unittest.mock import MagicMock, call
import sys
from pathlib import Path

# --- Dependency Mocking ---
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['sqlalchemy.orm'] = MagicMock()
sys.modules['sqlalchemy.ext.declarative'] = MagicMock()

# Setup explicit mocks for the models so we can verify class instantiation
class MockRecordMaterial:
    def __init__(self, **kwargs):
        self.data = kwargs
        self.material_id = kwargs.get('material_id')

    @classmethod
    def model_validate(cls, data):
        return cls(**data)

    def model_dump(self):
        return self.data

class MockRecordChem:
    def __init__(self, **kwargs):
        self.data = kwargs

    @classmethod
    def model_validate(cls, data):
        return cls(**data)

    def model_dump(self):
        return self.data

class MockRecordMaterialDB:
    # Class attributes need to be mocks to support .in_()
    material_id = MagicMock()

    def __init__(self, **kwargs):
        self.data = kwargs
        self.material_id = kwargs.get('material_id')
        # Populate attributes from kwargs for 'update' check
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockRecordChemDB:
    def __init__(self, **kwargs):
        self.data = kwargs

# Mock 'ingest.schemas'
mock_ingest = MagicMock()
mock_ingest.RecordMaterial = MockRecordMaterial
mock_ingest.RecordChem = MockRecordChem
sys.modules['ingest.schemas'] = mock_ingest

# Mock 'database.models'
mock_db_models = MagicMock()
mock_db_models.RecordMaterialDB = MockRecordMaterialDB
mock_db_models.RecordChemDB = MockRecordChemDB
mock_db_models.Base = MagicMock()
sys.modules['database.models'] = mock_db_models

# --- Import Function Under Test ---
class TestPopulateDBOptimization(unittest.TestCase):

    def setUp(self):
        # Setup Session Mock
        self.session = MagicMock()

        # Setup query behavior for existing materials
        # Default: no existing materials found
        self.query_mock = self.session.query.return_value
        self.filter_mock = self.query_mock.filter.return_value
        self.filter_mock.all.return_value = []

    def test_process_records_batches_inserts(self):
        from database.populate_db import process_records

        records = [
            {'material_id': 'm1', 'substance': 'Mat1'},
            {'substance': 'Chem1'},
            {'material_id': 'm2', 'substance': 'Mat2'},
            {'substance': 'Chem2'}
        ]

        process_records(self.session, records)

        # 1. Verify NO individual add/merge
        self.assertFalse(self.session.add.called, "Should not use session.add individually")
        self.assertFalse(self.session.merge.called, "Should not use session.merge individually")

        # 2. Verify bulk_save_objects was called
        self.assertTrue(self.session.bulk_save_objects.called, "Should use bulk_save_objects")

        # 3. Verify arguments to bulk_save_objects
        calls = self.session.bulk_save_objects.call_args_list
        saved_objects = []
        for c in calls:
            args, _ = c
            saved_objects.extend(args[0])

        chem_saved = [o for o in saved_objects if isinstance(o, MockRecordChemDB)]
        mat_saved = [o for o in saved_objects if isinstance(o, MockRecordMaterialDB)]

        self.assertEqual(len(chem_saved), 2)
        self.assertEqual(len(mat_saved), 2)

        self.assertTrue(self.filter_mock.all.called)

    def test_process_records_updates_existing(self):
        from database.populate_db import process_records

        # Setup existing record in DB
        existing_record = MockRecordMaterialDB(material_id='m1', substance='OldVal')
        self.filter_mock.all.return_value = [existing_record]

        records = [
            {'material_id': 'm1', 'substance': 'NewVal'}, # Should update
            {'material_id': 'm2', 'substance': 'Mat2'}    # Should insert
        ]

        process_records(self.session, records)

        # 1. Verify 'm1' was updated in place
        self.assertEqual(existing_record.substance, 'NewVal')

        # 2. Verify 'm2' was added to bulk save
        calls = self.session.bulk_save_objects.call_args_list
        saved_objects = []
        for c in calls:
            args, _ = c
            saved_objects.extend(args[0])

        mat_saved = [o for o in saved_objects if isinstance(o, MockRecordMaterialDB)]
        self.assertEqual(len(mat_saved), 1)
        self.assertEqual(mat_saved[0].material_id, 'm2')

if __name__ == '__main__':
    unittest.main()
