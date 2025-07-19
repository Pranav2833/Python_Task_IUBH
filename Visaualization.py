#THIS COMMENT IS NEW FEATURE

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from abc import ABC, abstractmethod
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

# === Paths ===
IDEAL_PATH = r"datasets/ideal.csv"
TRAIN_PATH = r"datasets/train.csv"
TEST_PATH = r"datasets/test.csv"

# === SQLAlchemy Setup ===
Base = declarative_base()
MYSQL_USER = 'root'
MYSQL_PASSWORD = quote_plus('Test@7890')
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_DB = 'python'
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
Session = sessionmaker(bind=engine)
session = Session()

class DataProcessor(ABC):
    """Abstract base class for loading data."""
    @abstractmethod
    def load_data(self):
        pass

class MappingError(Exception):
    """Custom exception for handling mapping errors."""
    pass

class DataHandler(DataProcessor):
    """Base class for data handling tasks."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load data using pandas."""
        self.data = pd.read_csv(self.file_path)

class FunctionMatcher(DataHandler):
    """Matches training functions to ideal functions using least squares."""
    def __init__(self, train_file, ideal_file):
        super().__init__(train_file)
        self.ideal_handler = DataHandler(ideal_file)
        self.train_data = None
        self.ideal_data = None
        self.best_matches = {}
        self.max_deviation_map = {}

    def load_data(self):
        super().load_data()
        self.train_data = self.data
        self.ideal_handler.load_data()
        self.ideal_data = self.ideal_handler.data

    def match_functions(self):
        for train_col in ['y1', 'y2', 'y3', 'y4']:
            y_train = self.train_data[train_col].values
            min_error = float('inf')
            best_col = None
            for i in range(1, 51):
                ideal_col = f'y{i}'
                error = np.sum((y_train - self.ideal_data[ideal_col].values) ** 2)
                if error < min_error:
                    min_error = error
                    best_col = ideal_col
            self.best_matches[train_col] = best_col
            max_dev = np.max(np.abs(y_train - self.ideal_data[best_col].values))
            self.max_deviation_map[best_col] = max_dev

    def get_results(self):
        return self.best_matches, self.max_deviation_map

class TestDataMapper(DataHandler):
    """Maps test data points to matched ideal functions within deviation threshold."""
    def __init__(self, test_file, ideal_data, match_results):
        super().__init__(test_file)
        self.ideal_data = ideal_data
        self.best_matches, self.max_deviation_map = match_results
        self.mapped_points = []

    def map_points(self):
        for _, row in self.data.iterrows():
            x, y_test = row['x'], row['y']
            mapped = False
            for train_col, ideal_col in self.best_matches.items():
                y_ideal = np.interp(x, self.ideal_data['x'], self.ideal_data[ideal_col])
                deviation = abs(y_test - y_ideal)
                threshold = self.max_deviation_map[ideal_col] * np.sqrt(2)
                if deviation <= threshold:
                    self.mapped_points.append({
                        'x(test function)': x,
                        'y(test function)': y_test,
                        'Delta Y (test function)': deviation,
                        'ideal_function': ideal_col
                    })
                    mapped = True
                    break
            if not mapped:
                self.mapped_points.append({
                    'x(test function)': x,
                    'y(test function)': y_test,
                    'Delta Y (test function)': None,
                    'ideal_function': None
                })

        if not self.mapped_points:
            raise MappingError("No test data could be mapped to ideal functions.")

    def get_mapped_df(self):
        return pd.DataFrame(self.mapped_points)

def create_table_class(name, columns, df):
    """Dynamically create a SQLAlchemy ORM table class."""
    attrs = {'__tablename__': name, 'id': Column(Integer, primary_key=True)}
    for col in columns:
        dtype = df[col].dtype
        if dtype in ['float64', 'int64']:
            attrs[col] = Column(Float)
        else:
            max_len = int(df[col].astype(str).map(len).max())
            attrs[col] = Column(String(max(20, max_len)))
    return type(name.capitalize(), (Base,), attrs)

def main():
    try:
        matcher = FunctionMatcher(TRAIN_PATH, IDEAL_PATH)
        matcher.load_data()
        matcher.match_functions()
        results = matcher.get_results()

        print("\nBest matching ideal functions:")
        for k, v in results[0].items():
            print(f"{k} => {v} | Max deviation: {results[1][v]:.6f}")

        mapper = TestDataMapper(TEST_PATH, matcher.ideal_data, results)
        mapper.load_data()
        mapper.map_points()
        mapped_df = mapper.get_mapped_df()

        datasets = {
            'training_data': matcher.train_data,
            'ideal_functions': matcher.ideal_data,
            'mapped_test_points': mapped_df
        }

        for table_name, df in datasets.items():
            df = df.replace({pd.NA: None, np.nan: None})
            df.columns = [col.strip() for col in df.columns]
            table_class = create_table_class(table_name, df.columns, df)
            Base.metadata.create_all(engine)
            rows = [table_class(**{k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in df.to_dict(orient='records')]
            session.bulk_save_objects(rows)
        session.commit()

        valid_mappings = mapped_df.dropna(subset=['ideal_function']).copy()
        valid_mappings['ideal_function'] = valid_mappings['ideal_function'].astype(str)
        function_labels = valid_mappings['ideal_function'].unique().tolist()

        source_mapped = ColumnDataSource(valid_mappings)
        color_mapper = factor_cmap('ideal_function', palette=Category10[10], factors=function_labels)

        output_file("combined_data_visualization.html")
        plot = figure(title="Combined Visualization: Test Data, Mappings, Deviations, and Line Comparisons",
                      x_axis_label="x", y_axis_label="y", width=1000, height=600)

        plot.scatter(x=mapped_df['x(test function)'],
                     y=mapped_df['y(test function)'],
                     size=6, color="darkred", alpha=0.4, legend_label="Test Data")

        plot.scatter(x="x(test function)", y="y(test function)", source=source_mapped,
                     size=6, color=color_mapper, alpha=0.8, legend_field="ideal_function")

        plot.vbar(x="x(test function)", top="Delta Y (test function)", width=0.3,
                  source=source_mapped, color=color_mapper, alpha=0.5, legend_field="ideal_function")

        for col in ['y1', 'y2', 'y3', 'y4']:
            y_train = matcher.train_data[col]
            best_match = results[0][col]
            plot.line(matcher.train_data['x'], y_train, color="blue", line_width=2, alpha=0.5, legend_label=f"Train {col}")
            plot.line(matcher.ideal_data['x'], matcher.ideal_data[best_match], color="green", line_dash="dashed",
                      line_width=2, alpha=0.5, legend_label=f"Ideal {best_match}")

        plot.legend.click_policy = "hide"
        plot.legend.location = "top_left"
        save(plot)
        print("\n✅ Visualization saved to combined_data_visualization.html")
        print("✅ All data successfully stored in MySQL database.")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

    finally:
        session.close()

if __name__ == "__main__":
    main()



