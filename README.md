# practice-
# PySpark Interview Questions — Solutions (Levels 1–3)

**Author:** Ludo (for Mahendra)

**Usage:** Run these snippets inside a PySpark shell (pyspark) or a script with a `SparkSession` available as `spark`.

---

## Level 1 — EASY (10) — Solutions

### 1. Read a CSV file with a header and infer schema

```python
df = spark.read.csv('/path/to/file.csv', header=True, inferSchema=True)
df.printSchema()
df.show(5)
```

### 2. Select specific columns from a DataFrame

```python
selected = df.select('id', 'name')
selected.show(5)
```

### 3. Filter rows where age > 25

```python
from pyspark.sql.functions import col
filtered = df.filter(col('age') > 25)
filtered.show(5)
```

### 4. Add a new column `bonus` = salary * 0.10

```python
from pyspark.sql.functions import expr
with_bonus = df.withColumn('bonus', col('salary') * 0.10)
with_bonus.select('salary', 'bonus').show(5)
```

### 5. Rename a column `emp_id` → `employee_id`

```python
renamed = df.withColumnRenamed('emp_id', 'employee_id')
renamed.printSchema()
```

### 6. Drop duplicates based on a single column

```python
deduped = df.dropDuplicates(['email'])
deduped.count()
```

### 7. Count total number of rows

```python
row_count = df.count()
print(row_count)
```

### 8. Sort employees by salary in descending order

```python
sorted_df = df.orderBy(col('salary').desc())
sorted_df.show(10)
```

### 9. Replace null values in a column with a default value

```python
filled = df.fillna({'phone': 'N/A', 'salary': 0})
filled.select('phone', 'salary').show(5)
```

### 10. Create DataFrame from a Python list of tuples

```python
data = [(1, 'Aarav', 30), (2, 'Isha', 25)]
columns = ['id', 'name', 'age']
local_df = spark.createDataFrame(data, schema=columns)
local_df.show()
```

---

## Level 2 — MEDIUM (10) — Solutions

### 11. Group by department and calculate total salary

```python
from pyspark.sql.functions import sum as _sum
agg_df = df.groupBy('department').agg(_sum('salary').alias('total_salary'))
agg_df.show()
```

### 12. Join two DataFrames: inner join on emp_id

```python
emp = spark.read.parquet('/path/emp.parquet')
salary = spark.read.parquet('/path/salary.parquet')
joined = emp.join(salary, emp.emp_id == salary.emp_id, how='inner')
joined.select(emp.emp_id, 'name', 'salary').show(5)
```

### 13. Window function — row_number() per department ordered by salary

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
w = Window.partitionBy('department').orderBy(col('salary').desc())
ranked = df.withColumn('rn', row_number().over(w))
ranked.filter(col('rn') <= 3).show()
```

### 14. Read a JSON file and flatten nested JSON

```python
nested = spark.read.json('/path/nested.json', multiLine=True)
# Example: explode a nested array column 'items' and select nested fields
from pyspark.sql.functions import explode
flat = nested.withColumn('item', explode(col('items')))
flat2 = flat.select('id', 'timestamp', 'item.*')
flat2.show(5)
```

### 15. Pivot operation: convert month-wise sales columns to rows

```python
sales = spark.createDataFrame([
    (1, '2025', 100, 120, 130),
], ['id', 'year', 'jan', 'feb', 'mar'])
# Unpivot using stack (SQL expression)
unpivoted = sales.selectExpr('id', 'year', "stack(3, 'jan', jan, 'feb', feb, 'mar', mar) as (month, sales)")
unpivoted.show()
```

### 16. Write a DataFrame to Parquet format partitioned by department

```python
df.write.mode('overwrite').partitionBy('department').parquet('/out/employee_parquet')
```

### 17. Use `when` + `otherwise` to create a derived column

```python
from pyspark.sql.functions import when
df2 = df.withColumn('seniority', when(col('experience') >= 5, 'Senior').otherwise('Junior'))
df2.select('name', 'experience', 'seniority').show(5)
```

### 18. Remove special characters from a string column using `regexp_replace`

```python
from pyspark.sql.functions import regexp_replace
clean = df.withColumn('clean_name', regexp_replace(col('name'), '[^A-Za-z0-9 ]', ''))
clean.select('name', 'clean_name').show(5)
```

### 19. Explain wide vs narrow transformations (example in code)

**Narrow example:** `filter`, `select` (no shuffle)

```python
narrow = df.filter(col('age') > 30).select('id', 'name')
```

**Wide example:** `groupBy`, `join` (causes shuffle)

```python
wide = df.groupBy('department').agg(_sum('salary'))
```

### 20. Union two DataFrames with identical schema

```python
u = df.unionByName(local_df)  # unionByName is safer
u.show(5)
```

---

## Level 3 — HARD (10) — Solutions

### 21. Solve skew join problem using salting technique (sketch + code)

When a few keys have huge partitions, add a random salt to small table and replicate large-key rows.

```python
from pyspark.sql.functions import lit, rand, floor
# small_df is small dimension, big_df is large fact
salted_small = small_df.withColumn('salt', floor(rand() * 10))
salted_big = big_df.withColumn('salt', floor(rand() * 10))
joined = salted_big.join(salted_small, ['join_key', 'salt'])
```

Adjust salt range (10) based on skew severity.

### 22. Optimize a job by using broadcast join when one table is small

```python
from pyspark.sql.functions import broadcast
joined = big_df.join(broadcast(small_df), 'emp_id')
```

Broadcast avoids shuffle of the big table.

### 23. Implement SCD Type-2 using PySpark (code)

This example assumes `source_df` contains new incoming records and `dim_df` is historical dimension with `effective_from`, `effective_to`, `is_current`.

```python
from pyspark.sql.functions import current_date
# key = 'emp_id', business attrs = ['name','dept']
new = source_df.alias('new')
old = dim_df.alias('old')
# Join on business key to find changes
compare = old.join(new, on='emp_id', how='right')
# For rows that changed, close old record and insert new record
# (Detailed production implementation uses left-anti / left-semi and union)
# Simple pattern (conceptual):
changed = new.join(dim_df, on='emp_id').filter(
    (new.name != dim_df.name) | (new.dept != dim_df.dept)
)
# Close existing records
to_close = dim_df.join(changed.select('emp_id'), on='emp_id').withColumn('effective_to', current_date()).withColumn('is_current', lit(False))
# Insert new versions
new_versions = changed.withColumn('effective_from', current_date()).withColumn('effective_to', lit(None)).withColumn('is_current', lit(True))
scd2 = dim_df.join(changed.select('emp_id'), on='emp_id', how='left_anti').unionByName(to_close).unionByName(new_versions)
```

(Production systems often use Delta MERGE for atomicity.)

### 24. Explain and implement Delta Lake MERGE operation

```sql
-- SQL style (Spark SQL with Delta Lake)
MERGE INTO dim AS target
USING updates AS source
ON target.emp_id = source.emp_id
WHEN MATCHED AND (target.name <> source.name OR target.dept <> source.dept)
  THEN UPDATE SET is_current = false, effective_to = current_date()
WHEN NOT MATCHED
  THEN INSERT (emp_id, name, dept, effective_from, effective_to, is_current) VALUES (source.emp_id, source.name, source.dept, current_date(), NULL, true)
```

### 25. Read data in batches from API using PySpark and create a streaming DataFrame (sketch)

Structured streaming typically reads from Kafka, files, sockets. For an API you'd implement a micro-batcher that writes to a directory or Kafka and then use Spark streaming to read.

```python
# pseudo: fetch API -> write JSON files into /stream/input -> spark.readStream.json('/stream/input')
```

### 26. Handling corrupt records in JSON using `badRecordsPath`

```python
spark.read.option('badRecordsPath', '/tmp/bad').json('/path/json')
```

Also `mode='PERMISSIVE'|'DROPMALFORMED'|'FAILFAST'` are useful.

### 27. Create a custom UDF and use it inside a transformation

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
def mask_email(e):
    if not e:
        return None
    parts = e.split('@')
    return parts[0][:2] + '***@' + parts[1]
mask_udf = udf(mask_email, StringType())
masked = df.withColumn('masked_email', mask_udf(col('email')))
masked.select('email', 'masked_email').show(5)
```

Note: Prefer `pyspark.sql.functions` native functions where possible for performance. Pandas UDFs (vectorized) are faster for heavy Python logic.

### 28. Explain and implement checkpointing in Structured Streaming

```python
stream = (
    spark.readStream.format('json').schema(schema).load('/stream/input')
    .writeStream.format('parquet')
    .option('checkpointLocation', '/stream/checkpoint')
    .start('/stream/output')
)
```

Checkpointing persists offsets/progress and is required for exactly-once/fault-tolerant streaming.

### 29. Perform an incremental load using watermark + append-only mode

```python
stream = (
    spark.readStream.format('parquet').schema(schema).load('/stream/input')
    .withWatermark('event_time', '1 hour')
    .groupBy('id', window(col('event_time'), '1 hour'))
    .agg(_sum('value'))
    .writeStream.outputMode('append').option('checkpointLocation', '/chk').start()
)
```

Watermarking allows state to be cleaned up for late data.

### 30. Optimize a DataFrame using caching, repartitioning, and coalesce

```python
# Cache
cached = df.cache()
cached.count()  # materialize
# Repartition before expensive shuffle-heavy operations (e.g., join)
repart = df.repartition(200, 'join_key')
# After writing, reduce small files by coalesce
repart.write.mode('overwrite').parquet('/out')
# When writing final output (few files) use coalesce(1) or coalesce(n)
```

---

## Quick tips (practical)

* Prefer DataFrame API over RDD for production code.
* Use broadcast join when one table < ~100MB (tune per cluster).
* Avoid UDFs where built-in SQL functions suffice.
* Profile with Spark UI and tune shuffle partitions (`spark.sql.shuffle.partitions`).
* For production SCD and CDC, prefer Delta Lake `MERGE` for atomicity.

---

If you want, I can:

* Convert this document to a PDF.
* Provide expanded explanations and test data for any question.
* Create runnable Jupyter notebooks for practice.

Tell me what you want next.
