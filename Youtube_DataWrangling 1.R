#install.packages("tidyverse")
#install.packages("tidyr")
#install.packages("dplyr")
library(dplyr)
library(tidyverse)
library(tidyr)


#reading the csv data
video_details_df<- read_csv(file="tedxChannelVideosDetails-13May.csv")
summary(video_details_df)
view(video_details_df)

# dropping "Favorite_Count" columns
video_details_df <- select(video_details_df, -Favorite_Count)
glimpse(video_details_df)

# Check data types of columns
column_types <- sapply(video_details_df, class)
print(column_types)

#checking null values
null_values <- is.na(video_details_df)
summary(null_values)

#Categorize published time and rename it to utc_day_part
utc_day_part <- function(Published_Time) {
  paste(
    c("Night", "Morning", "Afternoon", "Evening", "Night")[
      cut(as.numeric(format(Published_Time, "%H%M")), c(0, 530, 1159, 1700 ,2100, 2359))
    ]
  )
}

#adding the column at 3rd index
video_details_df <- cbind(video_details_df[, 1:2], utc_day_part=utc_day_part(video_details_df$Published_Time), video_details_df[, 3:ncol(video_details_df)])

#Extracting Day of Week from Published Time
Day_Of_Week <- weekdays(video_details_df$Published_Time)
video_details_df <- cbind(video_details_df[, 1:3], Day_Of_Week, video_details_df[, 4:ncol(video_details_df)])

#Extracting Minutes from Duration
minutes <- as.numeric(substr(video_details_df$Duration, 3, regexpr("M", video_details_df$Duration) - 1))
video_details_df <- cbind(video_details_df[, 1:9], minutes, video_details_df[, 10:ncol(video_details_df)])

#Pre-Processing Tags column
clean_tags<-function(tags){
tag_elements <- unlist(strsplit(tags, ","))
cleaned_elements <- tag_elements[-grep("TEDxTalks|\\[TEDxEID:\\d+\\]", tag_elements)]
cleaned_tag <- paste(cleaned_elements, collapse = ",")
return(cleaned_tag)

}
clean_tags(video_details_df$Tags)
video_details_df$Tags <- sapply(video_details_df$Tags, clean_tags)

view(video_details_df)