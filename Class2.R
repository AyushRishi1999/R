#session 2.1
install.packages(c("purrr","repurrrsive","doParallel","foreach"))
library(repurrrsive)
library(tidyverse)

view(iris)
avg = c()
for(i in 1:ncol(iris)){
  avg[i]  = mean(iris[ ,i])
}
avg

add_one <-function(num) {
  out <- tryCatch({
    num <- num+1
  },error=function(cond){
    NA
  })
  # be sure to match the input variable name
  return(out)
  # return() says what the output is
}
a=add_one("big data")
print(a)

add_two_values = function(num1=0,num2=0){
  if(is.numeric(num1)&is.numeric(num2)){
    s=num1+num2
  }
  else{
    s=NA
  }
  return(s)
}

add_two_values("1")

#MAP

map_dbl(c(3,4,5), sqrt)
map_dbl(1:3, function(x){x^2})

library(repurrrsive)
sw_people
sw_people[1]
sw_people[[1]]$name
sw_people[[1]]$mass
sw_people[[1]]$height #in cm
sw_people[[1]]$gender

mass=map_dbl(sw_people, function(x){as.numeric(x$mass)})
height=map_dbl(sw_people, function(x){as.numeric(x$height)})
name=map_chr(sw_people, function(x){x$name})
gender=map_chr(sw_people, function(x){x$gender})
bmi=mass/(height/100)^2
print(bmi)

bmi_category = function(v){
  if(is.na(v)){
    bmi_c = NA
  }
  else if(v<18.5){
    bmi_c = "underweight"
  }
  else if(v>=18.5 & v<=24.9){
    bmi_c="normalweight"
  }
  else if(v>=25 & v<=29.9){
    bmi_c="overweight"
  }
  else if(v>=30){
    bmi_c="obese"
  }
  return(bmi_c)
}

bmi_category=map_chr(bmi,bmi_category)

df = data.frame(
  name,
  gender,
  height,
  mass,
  bmi,
  bmi_category
)
view(df)

gh_users
followers = map_dbl(gh_users, ~.$followers)
followers

map_dbl(iris, mean)

as.Date(gh_users[[1]]$updated_at)-as.Date(gh_users[[1]]$created_at)
map(gh_users, ~as.Date(.$updated_at)-as.Date(.$created_at))

#MAP DF
map_df(
  gh_users,
  # for each element in the list
  magrittr::extract,
  # run the extract function in the magrittr package
  c("name","followers","following","public_repos")
  # values to extract()
)

#WALK
x=list(1,"a",3)
x |> walk(print)

library(foreach)
library(doParallel)
parallel::detectCores()

#Session 2.2

install.packages(c("httr","gh","devtools"))
library(httr)
library(gh)
library(tidyverse)

repo = GET(url = "https://api.github.com/users/hadley/repos")
content(repo)

response= GET(url = "https://api.github.com/orgs/youtube/repos")
con = content(response)
str(con)
length(con)
names(response)
map_chr(response, content.$name)
library(gh)
my_token = "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"
Sys.setenv(GITHUB_TOKEN = my_token)

hadley = gh("/users/hadley")
class(hadley)
length(hadley)
names(hadley)

hadley_repos = gh("/users/hadley/repos", .limit=Inf)
length(hadley_repos)

members = gh("/orgs/facebook/members", .limit=Inf)
login = map_chr(members, ~.$login)

member = gh("/users/{login}", login="aaronabramov")
followers = map_int(
  login, 
  function(login){
    member = gh("/users/{user}", user=login)
    return(member$followers)
  }
)
followers
df = data.frame(login,followers)
df





















