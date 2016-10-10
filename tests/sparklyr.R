library(sparklyr)
library(data.table)

### Install Spark locally for use by R. Create spark session. 
### You will need winutils.exe from https://github.com/steveloughran/winutils/raw/master/hadoop-2.6.0/bin/ ;)

### Needs to download some dependencies ! com.databricks#spark-csv_2.11 com.amazonaws#aws-java-sdk-pom added as a dependency

#spark_install()
sc <- spark_connect(master = "local", config = list())  # pb with dependencies, let's not use them

# Try to read data
spark_read_csv(sc, name = "test", path="file:///C:/workspace/dataAnalysis/Sephora/data/01-sephorafreuprod_2016-01-01.tsv", header = F, delimiter = "\t")

df <- fread("temp.tsv", sep = "\t")
df_tbl <- copy_to(sc, df)

# Analyse error 
spark_web(sc)

spark_disconnect(sc)
spark_disconnect_all()
