import findspark
findspark.init()

import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear
from pyspark.sql import types as t


# read config file
config = configparser.ConfigParser()
config.read('dl.cfg')

# configure aws access keys
os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    """
    Get or create an Spark session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .appName("Sparkify") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data, run_start_time):
    """
    Load json input data (song_data) from input_data path,
    process the data to extract song_table and artists_table, and
    store the queried data to parquet files.
    """
    ### Load song_data
    start_sd = datetime.now()
    start_sdl = datetime.now()
    
    print("Start processing song_data json files...")

    # read song data file
    print(f"Reading song_data files from {input_data}...")
    df_sd = spark.read.json(input_data)
    stop_sdl = datetime.now()
    print(f"finished in {stop_sdl - start_sdl}.")
    
    print("song_data schema:")
    df_sd.printSchema()

    ### Create and write songs_table
    # extract columns to create songs table
    start_st = datetime.now()
    df_sd.createOrReplaceTempView("songs_table")

    songs_table = spark.sql("""
                            select song_id, title, artist_id, year, duration
                            from songs_table
                            order by song_id
                            """)

    songs_table.printSchema()
    songs_table.show(5, truncate=False)

    # write songs table to parquet files partitioned by year and artist
    songs_table_path = f"{output_data}/songs-{run_start_time}"

    # write df to spark parquet file (partitioned by year and artist_id)
    print(f"writing songs_table parquet files to {songs_table_path}...")
    songs_table.write.mode("overwrite").partitionBy("year", "artist_id").parquet(songs_table_path)
    stop_st = datetime.now()
    print(f"finished in {stop_st - start_st}")

    ### Create and write artists_table
    # extract columns to create artists table
    start_at = datetime.now()
    df_sd.createOrReplaceTempView("artists_table")
    artists_table = spark.sql("""
        select  artist_id        as artist_id,
                artist_name      as name,
                artist_location  as location,
                artist_latitude  as latitude,
                artist_longitude as longitude
        from artists_table
        order by artist_id desc
    """)

    artists_table.printSchema()
    artists_table.show(5, truncate=False)

    # write artists table to parquet files
    artists_table_path = f"{output_data}/artists-{run_start_time}"

    print(f"writing artists_table parquet files to {artists_table_path}...")
    songs_table.write.mode("overwrite").parquet(artists_table_path)
    stop_at = datetime.now()
    print(f"finished in {stop_at - start_at}")
    
    stop_sd = datetime.now()
    print(f"finished processing in {stop_sd - start_sd}.\n")

    return songs_table, artists_table


def process_log_data(spark, input_data_ld, input_data_sd, output_data, run_start_time):
    """
    Load json input data (log_data) from input_data path,
    process the data to extract users_table, time_table,
    songplays_table, and store the queried data to parquet files.
    """
    ### Load log_data
    start_ld = datetime.now()
    print("start processing log_data json files...")

    # read log data file
    print(f"reading log_data files from {input_data_ld}...")
    df_ld = spark.read.json(input_data_ld)
    stop_ld = datetime.now()
    print(f"finished reading log_data in {stop_ld - start_ld}.")

    ### Create and write users_table
    # filter by actions for song plays
    start_ut = datetime.now()
    df_ld_filtered = df_ld.filter(df_ld.page == 'NextSong')

    # extract columns for users table
    df_ld_filtered.createOrReplaceTempView("users_table")
    users_table = spark.sql("""
                            select distinct userId    as user_id,
                                            firstName as first_name,
                                            lastName  as last_name,
                                            gender,
                                            level
                            from users_table
                            order by last_name
                            """)
    
    users_table.printSchema()
    users_table.show(5)

    # write users table to parquet files
    users_table_path = f"{output_data}/users-{run_start_time}"
    print(f"writing users_table parquet files to {users_table_path}...")
    users_table.write.mode("overwrite").parquet(users_table_path)
    stop_ut = datetime.now()
    print(f"finished in {stop_ut - start_ut}.")

    ### Create and write time_table
    # create timestamp column from original timestamp column
    start_tt = datetime.now()
    print("creating timestamp column...")
    @udf(t.TimestampType())
    def get_timestamp (ts):
        return datetime.fromtimestamp(ts / 1000.0)

    df_ld_filtered = df_ld_filtered.withColumn("timestamp", get_timestamp("ts"))
    
    df_ld_filtered.printSchema()
    df_ld_filtered.show(5)

    # create datetime column from original timestamp column
    print("creating datetime column...")
    @udf(t.StringType())
    def get_datetime(ts):
        return datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

    df_ld_filtered = df_ld_filtered.withColumn("datetime", get_datetime("ts"))
    
    df_ld_filtered.printSchema()
    df_ld_filtered.show(5)

    # extract columns to create time table
    df_ld_filtered.createOrReplaceTempView("time_table")
    time_table = spark.sql("""
                            select distinct datetime                as start_time,
                                            hour(timestamp)         as hour,
                                            day(timestamp)          as day,
                                            weekofyear(timestamp)   as week,
                                            month(timestamp)        as month,
                                            year(timestamp)         as year,
                                            dayofweek(timestamp)    as weekday
                            from time_table
                            order by start_time
                            """)

    time_table.printSchema()
    time_table.show(5)

    # write time table to parquet files partitioned by year and month
    time_table_path = f"{output_data}/time-{run_start_time}"

    print(f"writing time_table parquet files to {time_table_path}...")
    time_table.write.mode("overwrite").partitionBy("year", "month").parquet(time_table_path)
    stop_tt = datetime.now()
    print(f"finished in {stop_tt - start_tt}.")

    ### Create and write songplays_table
    # read in song data to use for songplays table
    start_spt = datetime.now()
    song_data = input_data_sd
    print(f"reading song_data files from {song_data}...")
    df_sd = spark.read.json(song_data)

    # join log_data and song_data dataframes
    print("joining log_data and song_data dataframes...")
    df_ld_sd_joined = df_ld_filtered\
        .join(df_sd, (df_ld_filtered.artist == df_sd.artist_name) & \
                     (df_ld_filtered.song == df_sd.title))
    print("finished joined dataframe.")

    df_ld_sd_joined.printSchema()
    df_ld_sd_joined.show(5)

    # extract columns from joined song and log datasets
    # to create songplays table
    print("extracting columns from joined dataframe...")
    df_ld_sd_joined = df_ld_sd_joined.withColumn("songplay_id", \
                        monotonically_increasing_id())
    df_ld_sd_joined.createOrReplaceTempView("songplays_table")
    songplays_table = spark.sql("""
                                select songplay_id as songplay_id,
                                       timestamp   as start_time,
                                       userId      as user_id,
                                       level       as level,
                                       song_id     as song_id,
                                       artist_id   as artist_id,
                                       sessionId   as session_id,
                                       location    as location,
                                       userAgent   as user_agent
                                from songplays_table
                                order by (user_id, session_id)
                                """)

    songplays_table.printSchema()
    songplays_table.show(5, truncate=False)

    # write songplays table to parquet files partitioned by year and month
    songplays_table_path = f"{output_data}/songplays-{run_start_time}"

    print("writing songplays_table parquet files to {songplays_table_path}...")
    time_table.write.mode("overwrite").partitionBy("year", "month")\
            .parquet(songplays_table_path)
    stop_spt = datetime.now()
    print(f"finished in {stop_spt - start_spt}.")

    return users_table, time_table, songplays_table


def query_table_count(spark, table):
    """
    Query example returning row count of the given table.
    """
    return table.count()


def query_songplays_table(  spark, \
                            songs_table, \
                            artists_table, \
                            users_table, \
                            time_table, \
                            songplays_table):
    """
    Query example using all the created tables.
    Provides example set of songplays and who listened them.
    """
    df_all_tables_joined = songplays_table.alias('sp')\
        .join(users_table.alias('u'), col('u.user_id') \
                                    == col('sp.user_id'))\
        .join(songs_table.alias('s'), col('s.song_id') \
                                    == col('sp.song_id'))\
        .join(artists_table.alias('a'), col('a.artist_id') \
                                    == col('sp.artist_id'))\
        .join(time_table.alias('t'), col('t.start_time') \
                                    == col('sp.start_time'))\
        .select('sp.songplay_id', 'u.user_id', 's.song_id', 'u.last_name', \
                'sp.start_time', 'a.name', 's.title')\
        .sort('sp.start_time')\
        .limit(100)

    df_all_tables_joined.printSchema()
    df_all_tables_joined.show()

    return


def query_examples( spark, \
                    songs_table, \
                    artists_table, \
                    users_table, \
                    time_table, \
                    songplays_table):
    """
    Query example using all the created tables.
    """
    print(f"song_table count: {str(query_table_count(spark, songs_table))}")
    print(f"artists_table count: {str(query_table_count(spark, artists_table))}")
    print(f"users_table count: {str(query_table_count(spark, users_table))}")
    print(f"time_table count: {str(query_table_count(spark, time_table))}")
    print(f"songplays_table count: {str(query_table_count(spark, songplays_table))}")

    query_songplays_table(  spark, \
                            songs_table, \
                            artists_table, \
                            users_table, \
                            time_table, \
                            songplays_table)


def main():
    """
    Load json input data (song_data and log_data) from input_data path,
    process the data to extract songs_table, artists_table,
    users_table, time_table, songplays_table,
    and store the queried data to parquet files to output_data path.
    """
    run_start_time = datetime.now()
    print(f"\npipeline started at {run_start_time}\n")

    spark = create_spark_session()

    # uncomment to execute etl in aws s3
    #input_data_song = config['AWS']['INPUT_DATA_SONG']
    #input_data_log = config['AWS']['INPUT_DATA_LOG']
    #output_data = config['AWS']['OUTPUT_DATA']

    # uncomment to execute etl in local mode
    input_data_song = config['LOCAL']['INPUT_DATA_SONG']
    input_data_log = config['LOCAL']['INPUT_DATA_LOG']
    output_data = config['LOCAL']['OUTPUT_DATA']

    songs_table, artists_table = process_song_data( spark, \
                                                    input_data_song, \
                                                    output_data, \
                                                    run_start_time)
    
    users_table, time_table, songplays_table = process_log_data(spark, \
                                                                input_data_log, \
                                                                input_data_song, \
                                                                output_data, \
                                                                run_start_time)
    
    run_end_time = datetime.now()
    print(f"finished pipeline processing in {run_end_time - run_start_time}.")

    print("running queries...")
    query_examples( spark, \
                    songs_table, \
                    artists_table, \
                    users_table, \
                    time_table, \
                    songplays_table)

if __name__ == "__main__":
    main()
