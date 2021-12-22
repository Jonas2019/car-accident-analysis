import sys
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from data_filtering import *

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
    StructField('Amenity', BooleanType()),
    StructField('Bump', BooleanType()),
    StructField('Crossing', BooleanType()),
    StructField('Give_Way', BooleanType()),
    StructField('Junction', BooleanType()),
    StructField('No_Exit', BooleanType()),
    StructField('Railway', BooleanType()),
    StructField('Roundabout', BooleanType()),
    StructField('Station', BooleanType()),
    StructField('Stop', BooleanType()),
    StructField('Traffic_Calming', BooleanType()),
    StructField('Traffic_Signal', BooleanType()),
    StructField('Turning_Loop', BooleanType()),
    StructField('Sunrise_Sunset', StringType()),
    StructField('Civil_Twilight', StringType()),
    StructField('Nautical_Twilight', StringType()),
    StructField('Astronomical_Twiligh', StringType()),
])


def dataTraining(train, val, pipeline, evaluator):
    model = pipeline.fit(train)
    result_train = model.transform(train)
    result_val = model.transform(val)
    train_acc = evaluator.evaluate(result_train)
    val_acc = evaluator.evaluate(result_val)
    print('Traning accuracy:', train_acc)
    print('Validation accuracy:', val_acc)
    return model, result_train, result_val


def getResultByPredition(predict_result):
    sev1_pr = predict_result.filter(predict_result['Severity_pr']==1) \
              .groupBy('Severity').count().orderBy('Severity')
    sev2_pr = predict_result.filter(predict_result['Severity_pr']==2) \
              .groupBy('Severity').count().orderBy('Severity')
    sev3_pr = predict_result.filter(predict_result['Severity_pr']==3) \
              .groupBy('Severity').count().orderBy('Severity')
    return sev1_pr, sev2_pr, sev3_pr


def main():
    data = spark.read.csv('Accident_No_NA.csv',
        schema=accidents_schema, header=True)

    data = dataFiltering(data)
    columns = ['Severity', 'Start_Time',
               'Start_Lat', 'Start_Lng',
               'Temperature(F)', 'Pressure(in)',
               'Humidity(%)', 'Visibility(mi)',
               'Junction', 'Traffic_Signal']

    data = data.select(columns) \
           .filter(data['Temperature(F)']>-900) \
           .filter(data['Humidity(%)']>-900) \
           .filter(data['Pressure(in)']>-900) \
           .filter(data['Visibility(mi)']>-900)

    data = data.withColumn('Year', functions.year(data['Start_Time'])) \
               .withColumn('Month', functions.month(data['Start_Time'])) \
               .withColumn('DayofWeek', functions.dayofweek(data['Start_Time'])) \
               .withColumn('Start_hour', functions.hour(data['Start_Time']))
    data = data.cache()

    # str_indexer = StringIndexer(inputCol='Weather', outputCol='Weather_idx')

    encoder = OneHotEncoder(
        inputCols=['Month', 'DayofWeek', 'Start_hour'],
        outputCols=['MonthVec', 'DayofWeekVec', 'Start_hourVec'])

    features = ['Year', 'MonthVec', 'DayofWeekVec',
                'Start_hourVec', 'Start_Lat', 'Start_Lng',
                'Temperature(F)', 'Pressure(in)', 'Humidity(%)',
                'Visibility(mi)', 'Junction', 'Traffic_Signal']

    vec_assembler = VectorAssembler(
        inputCols=features,
        outputCol='features')

    dt_classifier = DecisionTreeClassifier(labelCol='Severity', maxDepth=15)

    pipeline = Pipeline(
        stages=[encoder, vec_assembler, dt_classifier])

    evaluator = MulticlassClassificationEvaluator(
        labelCol='Severity',
        predictionCol='prediction',
        metricName='accuracy')
    
    train, validation = data.randomSplit([0.8, 0.2])
    train = train.cache()
    validation = validation.cache()

    dt_model, dt_train, dt_val = dataTraining(
        train, validation, pipeline, evaluator)
    prediction_result = dt_val.withColumn(
        'Severity_pr', dt_val['prediction'].cast(IntegerType()))
    prediction_result = prediction_result.select(
        'Severity', 'Start_Lat', 'Start_Lng', 'Severity_pr').cache()

    sev1, sev2, sev3 = getResultByPredition(prediction_result)

    data_sample = data.sampleBy(
        'Severity',
        fractions={1: 0.2, 2: 0.9, 3:1},
        seed=10
    ).cache()
    train_sample, val_sample = data_sample.randomSplit([0.8, 0.2])
    train_sample = train_sample.cache()
    val_sample = val_sample.cache()

    new_model, new_dt_train, new_dt_val = dataTraining(
        train_sample, val_sample, pipeline, evaluator)
    new_prediction = new_dt_val.withColumn(
        'Severity_pr',
        new_dt_val['prediction'].cast(IntegerType()))
    new_prediction = new_prediction.select(
        'Severity', 'Start_Lat', 'Start_Lng', 'Severity_pr').cache()

    new_sev1, new_sev2, new_sev3 = getResultByPredition(new_prediction)

    # new_prediction.write.format('mongo').mode('overwrite').option(
    #     'spark.mongodb.output.uri',
    #     'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.dtPrediction'
    #     ).save()
    
    # new_sev1.write.format('mongo').mode('overwrite').option(
    #     'spark.mongodb.output.uri',
    #     'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Sev1Prediction'
    #     ).save()

    # new_sev2.write.format('mongo').mode('overwrite').option(
    #     'spark.mongodb.output.uri',
    #     'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Sev2Prediction'
    #     ).save()

    # new_sev3.write.format('mongo').mode('overwrite').option(
    #     'spark.mongodb.output.uri',
    #     'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Sev3Prediction'
    #     ).save()

    # new_model.save('DecisionTreePipelineModel')

if __name__ == '__main__':
    spark = SparkSession.builder.appName('machine learning').getOrCreate()

    assert spark.version >= '3.0'
    spark = SparkSession\
        .builder\
        .master('local[2]')\
        .appName('accidents_etl')\
        .config("spark.mongodb.input.uri", 'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Project')\
        .config('spark.mongodb.output.uri', 'mongodb+srv://dbAdmin:cmpt732@cluster732.jfbfw.mongodb.net/CMPT732.Project')\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    main()
