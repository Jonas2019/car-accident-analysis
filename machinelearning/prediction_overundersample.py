import pyspark
from pyspark.sql import SparkSession  
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession\
        .builder\
        .master('local[2]')\
        .appName('accidents_etl')\
        .config("spark.mongodb.input.uri", 'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Project')\
        .config('spark.mongodb.output.uri', 'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Project')\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()

assert spark.version >= '3.0'  # make sure we have Spark 3.0+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

accidents_schema = StructType([
    StructField('ID', StringType()),
    StructField('Severity', IntegerType()),
    StructField('Start_Time', TimestampType()),
    StructField('End_Time', TimestampType()),
    StructField('Start_Lat', DoubleType()),
    StructField('Start_Lng', DoubleType()),
    StructField('End_Lat', DoubleType()),
    StructField('End_Lng', DoubleType()),
    StructField('Distance(mi)', DoubleType()),
    StructField('Description', StringType()),
    StructField('Number', StringType()),
    StructField('Street', StringType()),
    StructField('Side', StringType()),
    StructField('City', StringType()),
    StructField('County', StringType()),
    StructField('State', StringType()),
    StructField('Zipcode', StringType()),
    StructField('Country', StringType()),
    StructField('Timezone', StringType()),
    StructField('Airport_Code', StringType()),
    StructField('Weather_Timestamp', TimestampType()),
    StructField('Temperature(F)', DoubleType()),
    StructField('Wind_Chill(F)', DoubleType()),
    StructField('Humidity(%)', DoubleType()),
    StructField('Pressure(in)', DoubleType()),
    StructField('Visibility(mi)', DoubleType()),
    StructField('Wind_Direction', StringType()),
    StructField('Wind_Speed(mph)', DoubleType()),
    StructField('Precipitation(in)', DoubleType()),
    StructField('Weather_Condition', StringType()),
    StructField('Amenity', StringType()),
    StructField('Bump', StringType()),
    StructField('Crossing', StringType()),
    StructField('Give_Way', StringType()),
    StructField('Junction', StringType()),
    StructField('No_Exit', StringType()),
    StructField('Railway', StringType()),
    StructField('Roundabout', StringType()),
    StructField('Station', StringType()),
    StructField('Stop', StringType()),
    StructField('Traffic_Calming', StringType()),
    StructField('Traffic_Signal', StringType()),
    StructField('Turning_Loop', StringType()),
    StructField('Sunrise_Sunset', StringType()),
    StructField('Civil_Twilight', StringType()),
    StructField('Nautical_Twilight', StringType()),
    StructField('Astronomical_Twiligh', StringType()),
])

df_load = spark.read.csv('Accident_No_NA.csv', schema=accidents_schema,header=True)

df_filter = df_load.filter(year(df_load['Start_Time'])!=2020).withColumn('Severity', df_load['Severity']-1)
columns = ['Severity','Start_Time','Start_Lat', 'Start_Lng','Temperature(F)', 'Pressure(in)',
           'Humidity(%)', 'Junction','Traffic_Signal']
df_filter = df_filter.select(columns)
df_filter = df_filter.filter(df_filter['Temperature(F)']> -999.0).filter(df_filter['Pressure(in)']> -999.0).filter(df_filter['Humidity(%)']>-999.0)

df_filter  = df_filter.withColumn('Year', year(df_filter ['Start_Time'])).withColumn('Month', month(df_filter['Start_Time'])).withColumn('Weekday', ((dayofweek(df_filter['Start_Time'])+5)%7)+1).withColumn('Hour', hour(df_load['Start_Time']))
df_ml = df_filter.cache()

#Undersampling
df_ml= df_ml.sampleBy('Severity',fractions={1: 0.4, 2: 1, 3:1},seed=10)

#Oversampling
df_1 = df_ml.filter(df_ml['Severity'] == 1.0)
df_2 = df_ml.filter(df_ml['Severity'] == 2.0)
df_3 = df_ml.filter(df_ml['Severity'] == 3.0)

count_1 = df_1.count()
count_2 = df_2.count() 
count_3 = df_3.count()
ratio_2 = count_1 / count_2
ratio_3 = count_1 / count_3

df_2_overampled = df_2.sample(withReplacement=True, fraction=ratio_2, seed=10)
df_3_overampled = df_3.sample(withReplacement=True, fraction=ratio_3, seed=10)
df_ml = df_1.unionAll(df_2_overampled).unionAll(df_3_overampled).cache()


indexers = [StringIndexer(inputCol="Junction", outputCol="Junction_Index"),StringIndexer(inputCol="Traffic_Signal", outputCol="Traffic_Signal_Index")]
pipeline = Pipeline(stages=indexers)
df_ml = pipeline.fit(df_ml).transform(df_ml)

encoder = [OneHotEncoder(inputCols=['Month', 'Weekday', 'Hour', 'Junction_Index',"Traffic_Signal_Index"], outputCols=['Month_ml', 'Weekday_ml', 'Hour_ml', 'Junction_ml',"Traffic_Signal_ml"])]
pipeline = Pipeline(stages=encoder)
df_ml = pipeline.fit(df_ml).transform(df_ml)

df_train, df_test = df_ml.randomSplit([0.8, 0.2])
df_train = df_train.cache()
df_test = df_test.cache()

assembler = VectorAssembler(inputCols=['Year', 'Month_ml', 'Weekday_ml', 'Hour_ml','Start_Lat', 'Start_Lng','Temperature(F)','Pressure(in)', 'Humidity(%)', 'Junction_ml','Traffic_Signal_ml'], outputCol='features')
model_reg =  DecisionTreeClassifier(featuresCol='features', labelCol='Severity', maxDepth=15)
pipeline = Pipeline(stages=[assembler, model_reg])
model = pipeline.fit(df_train)
pred_results = model.transform(df_test)
pred_train = model.transform(df_train)


evaluator_test = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction")
accuracy = evaluator_test.evaluate(pred_results)
print("Test Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

evaluator_train = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction")
accuracy = evaluator_train.evaluate(pred_train)
print("Train Accuracy = %s" % (accuracy))
print("Train Error = %s" % (1.0 - accuracy))

df_pred_results = pred_results['Start_Lat', 'Start_Lng', 'prediction','Severity']
df_pred_results = df_pred_results.withColumnRenamed('prediction', 'Pred_Severity')
df_pred_results = df_pred_results.withColumn('Accuracy', lit(accuracy))

pred_s1= df_pred_results.filter(df_pred_results['Pred_Severity']==1.0).groupBy('Severity').count().orderBy('Severity')
pred_s2 = df_pred_results.filter(df_pred_results['Pred_Severity']==2.0).groupBy('Severity').count().orderBy('Severity')
pred_s3 = df_pred_results.filter(df_pred_results['Pred_Severity']==3.0).groupBy('Severity').count().orderBy('Severity')


df_pred_results.write.format('mongo').mode('overwrite').option('spark.mongodb.output.uri',
        'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.PredResults_overunder').save()
pred_s1.write.format('mongo').mode('overwrite').option('spark.mongodb.output.uri',
        'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.PredS1_overunder').save()
pred_s2.write.format('mongo').mode('overwrite').option('spark.mongodb.output.uri',
        'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.PredS2_overunder').save()
pred_s3.write.format('mongo').mode('overwrite').option('spark.mongodb.output.uri',
        'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.PredS3_overunder').save()


#model.save('OverUnderSamplePipelineModel')
