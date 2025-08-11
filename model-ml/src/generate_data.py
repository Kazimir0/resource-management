import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data(num_rows=20000):
    """
    Generate a synthetic dataset for testing with the same structure
    as the original Kaggle medical appointments dataset.

    Args:
        num_rows (int): Number of rows to generate.

    Returns:
        pd.DataFrame: A pandas DataFrame with the generated test data.
    """
    # Define possible values for categorical columns
    genders = ['M', 'F']
    neighbourhoods = [
        'JARDIM CAMBURI', 'MARIA ORTIZ', 'JARDIM DA PENHA', 'SANTA MARTHA',
        'CENTRO', 'PRAIA DO SUÁ', 'ITARARÉ', 'SANTO ANTÔNIO', 'TABUAZEIRO',
        'BONFIM', 'CONSOLAÇÃO', 'ILHA DO PRÍNCIPE'
    ]
    no_show_status = ['No', 'Yes']

    # Generate base data
    data = {
        'PatientId': np.random.randint(int(1e12), int(1e14), size=num_rows, dtype=np.int64),
        'AppointmentID': np.arange(8000000, 8000000 + num_rows),
        'Gender': np.random.choice(genders, size=num_rows, p=[0.35, 0.65]),
        'Age': np.random.randint(0, 101, size=num_rows),
        'Neighbourhood': np.random.choice(neighbourhoods, size=num_rows),
        'Scholarship': np.random.randint(0, 2, size=num_rows),
        'Hipertension': np.random.randint(0, 2, size=num_rows),
        'Diabetes': np.random.randint(0, 2, size=num_rows),
        'Alcoholism': np.random.randint(0, 2, size=num_rows),
        'Handcap': np.random.choice([0, 1, 2, 3, 4], size=num_rows, p=[0.97, 0.02, 0.005, 0.003, 0.002]),
        'SMS_received': np.random.randint(0, 2, size=num_rows),
        'No-show': np.random.choice(no_show_status, size=num_rows, p=[0.8, 0.2])
    }

    # Generate date fields
    scheduled_dates = []
    appointment_dates = []
    base_date = datetime(2025, 10, 1) # Use future dates to ensure they are new

    for _ in range(num_rows):
        # Scheduling day can be at any time
        schedule_offset_days = np.random.randint(0, 60)
        schedule_offset_seconds = np.random.randint(0, 86400)
        scheduled_date = base_date + timedelta(days=int(schedule_offset_days), seconds=int(schedule_offset_seconds))
        scheduled_dates.append(scheduled_date.strftime('%Y-%m-%dT%H:%M:%SZ'))

        # Appointment day is on or after the scheduling day, with time reset
        wait_days = np.random.randint(0, 30)
        appointment_date = scheduled_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=wait_days)
        appointment_dates.append(appointment_date.strftime('%Y-%m-%dT%H:%M:%SZ'))

    data['ScheduledDay'] = scheduled_dates
    data['AppointmentDay'] = appointment_dates

    # Create and reorder the DataFrame to match the original
    df = pd.DataFrame(data)
    column_order = [
        'PatientId', 'AppointmentID', 'Gender', 'ScheduledDay', 'AppointmentDay',
        'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'Handcap', 'SMS_received', 'No-show'
    ]
    df = df[column_order]

    return df

# --- Main execution ---
if __name__ == '__main__':
    # Define the path for the new test file
    TEST_DATA_PATH = "test_data.csv"
    
    # Generate data
    print("Generating new test data...")
    test_dataframe = create_test_data(num_rows=20000)

    # Save the data to the specified CSV file
    try:
        test_dataframe.to_csv(TEST_DATA_PATH, index=False)
        print(f"File '{TEST_DATA_PATH}' was created successfully!")
        print(f"Number of generated records: {len(test_dataframe)}")
        print("\nFirst 5 rows of the generated dataset:")
        print(test_dataframe.head())
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")