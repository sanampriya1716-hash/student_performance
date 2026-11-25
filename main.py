
# ==========================================================
# PROJECT: STUDENT PERFORMANCE ANALYSIS
# Libraries: NumPy, Pandas, Matplotlib
# ==========================================================

# STEP 1: Import the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# STEP 2: Set up project paths

# Get the folder where this Python file is stored.
PROJECT_DIR = Path(__file__).resolve().parent

# Create an output folder for results and charts.
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Read the CSV file from the project folder.
DATA_FILE = PROJECT_DIR / "students.csv"

df = pd.read_csv(DATA_FILE)


# STEP 3: Display the original student data

print("\n========== STUDENT DATA ==========")
print(df.to_string(index=False))

print("\nDataset shape:", df.shape)
print("Number of students:", len(df))


# STEP 4: Define the subject columns

subjects = [
    "Maths",
    "Science",
    "English",
    "Computer",
    "Social_Science"
]

# Check that the required columns exist.
required_columns = ["Student_ID", "Name"] + subjects

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in students.csv: {missing_columns}"
    )

# Convert marks into numeric values.
# Invalid marks become NaN (missing values).
for subject in subjects:
    df[subject] = pd.to_numeric(
        df[subject], errors="coerce"
    )

# This project requires a complete set of valid marks.
if df[subjects].isna().any().any():
    raise ValueError(
        "Some subject marks are missing or non-numeric."
    )

# Marks must be between 0 and 100.
if not df[subjects].apply(
    lambda column: column.between(0, 100).all()
).all():
    raise ValueError(
        "All subject marks must be between 0 and 100."
    )


# STEP 5: Calculate total marks

# axis=1 means add marks across each student's row.
df["Total"] = df[subjects].sum(axis=1)

print("\n========== TOTAL MARKS ==========")
print(df[["Name", "Total"]].to_string(index=False))


# STEP 6: Calculate average and percentage

# There are five subjects, each out of 100.
df["Average"] = df[subjects].mean(axis=1)

df["Percentage"] = (
    df["Total"] / (len(subjects) * 100)
) * 100

print("\n========== AVERAGE AND PERCENTAGE ==========")
print(
    df[["Name", "Average", "Percentage"]]
    .round(2)
    .to_string(index=False)
)


# STEP 7: Assign grades

def get_grade(percentage):
    """Return a grade based on the student's percentage."""

    if percentage >= 90:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 70:
        return "B"
    elif percentage >= 60:
        return "C"
    elif percentage >= 50:
        return "D"
    else:
        return "F"


df["Grade"] = df["Percentage"].apply(get_grade)

print("\n========== STUDENT GRADES ==========")
print(df[["Name", "Percentage", "Grade"]].to_string(index=False))


# STEP 8: Calculate student ranks

# The highest total receives rank 1.
# Equal totals receive the same rank.
df["Rank"] = (
    df["Total"]
    .rank(method="min", ascending=False)
    .astype(int)
)

ranked_students = df.sort_values(
    ["Rank", "Name"]
)

print("\n========== STUDENT RANKING ==========")
print(
    ranked_students[
        ["Rank", "Name", "Total", "Percentage", "Grade"]
    ].round(2).to_string(index=False)
)


# STEP 9: Determine pass or fail

# Example rule: at least 35 marks in every subject.
# This is a sample rule and can be changed.
df["Result"] = np.where(
    (df[subjects] >= 35).all(axis=1),
    "Pass",
    "Fail"
)

print("\n========== PASS / FAIL ==========")
print(df[["Name", "Result"]].to_string(index=False))


# STEP 10: Find the class topper

# idxmax() returns the index of the highest total.
topper_index = df["Total"].idxmax()

# loc[] retrieves the complete row for that student.
topper = df.loc[topper_index]

print("\n========== CLASS TOPPER ==========")
print("Name:", topper["Name"])
print("Total:", topper["Total"], "out of 500")
print("Percentage:", round(topper["Percentage"], 2), "%")
print("Grade:", topper["Grade"])


# STEP 11: Perform subject-wise analysis

# Mean, maximum and minimum for each subject.
subject_average = df[subjects].mean()
subject_highest = df[subjects].max()
subject_lowest = df[subjects].min()

subject_summary = pd.DataFrame({
    "Average": subject_average,
    "Highest": subject_highest,
    "Lowest": subject_lowest
})

print("\n========== SUBJECT ANALYSIS ==========")
print(subject_summary.round(2))


# STEP 12: Identify students needing improvement

# Example rule: average marks below 50.
needs_improvement = df[df["Average"] < 50]

print("\n========== NEEDS IMPROVEMENT ==========")

if needs_improvement.empty:
    print("No students meet this condition.")
else:
    print(
        needs_improvement[["Name", "Average"]]
        .round(2)
        .to_string(index=False)
    )


# STEP 13: Calculate statistics using NumPy

# Convert all subject marks into a NumPy array.
marks_array = df[subjects].to_numpy()

print("\n========== NUMPY STATISTICS ==========")
print("Overall mean:", round(np.mean(marks_array), 2))
print("Median:", round(np.median(marks_array), 2))
print("Standard deviation:", round(np.std(marks_array), 2))
print("Highest mark:", np.max(marks_array))
print("Lowest mark:", np.min(marks_array))

# Percentile helps us understand score distribution.
print(
    "75th percentile:",
    round(np.percentile(marks_array, 75), 2)
)


# STEP 14: Calculate the grade distribution

grade_counts = (
    df["Grade"]
    .value_counts()
    .reindex(["A+", "A", "B", "C", "D", "F"], fill_value=0)
)

print("\n========== GRADE DISTRIBUTION ==========")
print(grade_counts)


# STEP 15: Create chart 1 - Average marks by subject


plt.figure(figsize=(9, 5))

subject_average.plot(
    kind="bar",
    color="skyblue",
    edgecolor="black"
)


plt.title("Average Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Marks out of 100")
plt.xticks(rotation=30)
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "subject_average.png", dpi=150)
plt.show()


# STEP 16: Create chart 2 - Student performance

df_sorted = df.sort_values("Total", ascending=False)

plt.figure(figsize=(9, 5))

plt.bar(df_sorted["Name"], df_sorted["Total"])

plt.title("Total Marks by Student")
plt.xlabel("Student")
plt.ylabel("Total Marks out of 500")
plt.ylim(0, 500)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "student_performance.png", dpi=150)
plt.show()


# STEP 17: Create chart 3 - Distribution of marks

plt.figure(figsize=(9, 5))

plt.hist(
    marks_array.flatten(),
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Subject Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Scores")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "marks_distribution.png", dpi=150)
plt.show()


# STEP 18: Create chart 4 - Grade distribution

plt.figure(figsize=(7, 5))

grade_counts.plot(kind="pie", autopct="%1.1f%%")

plt.title("Student Grade Distribution")
plt.ylabel("")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "grade_distribution.png", dpi=150)
plt.show()


# STEP 19: Save analysis results to CSV files

# Save complete student results.
df.to_csv(
    OUTPUT_DIR / "student_results.csv",
    index=False
)

# Save subject-wise statistics.
subject_summary.to_csv(
    OUTPUT_DIR / "subject_summary.csv"
)

# Save grade counts.
grade_counts.to_csv(
    OUTPUT_DIR / "grade_distribution.csv",
    header=["Number_of_Students"]
)


# STEP 20: Print completion message

print("\n========== PROJECT COMPLETED ==========")
print("All analysis results have been saved to:")
print(OUTPUT_DIR)