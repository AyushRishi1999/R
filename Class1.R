class(0.3)

paste("Artificial", "Intelligence")

FALSE==F

msis_concentrations<- factor(
  c("Big Data", "Digital Innovation", "Cybersecurity")
)
msis_concentrations

x=2
y<-3

f_name="Ayush"
l_name="Rishi"
length_f_name=nchar(f_name)
length_l_name=nchar(l_name)
paste(f_name,l_name)
length_f_name*length_r_name
length_f_name/length_r_name
length_f_name>length_l_name

rep("AI",3)

#Vector
x<-c(10,20,7,13)
names(x)<-c("value1","value2","value3","value4")
x
vec<-c("value1"=10,"value"=20, "value3"=30, "value"=40)

vec1<-c(1,3,5)
vec2<-c(11,13,15)
vec3<-c(vec1,vec2,c(21,23,25))

u<-c(10,20)
v<-c(1,2,3,4,5)
u+v

x<-c(10,20,30)
20 %in% x

basket <- c("apple","bananas")
"apple" %in% basket

vec=c(1,4,NA,2)
vec
sum(vec)
max(vec)
sum(vec,na.rm = T)
max(vec,na.rm = T)
vec5<-c("value1"=10,"value"=20, "value3"=30, "value"=40)
vec5[1]
x=list("Bob",c(100,80,90))
x
x=list(name="Bob",grades=c(100,80,90))
x[2] #returns a list
x["grades"]
class(x[2])
x[[2]] #returns a vector
class(x[[2]])
x$grades #returns the original vector

l=list(
  name=c("Alice", "Bob", "Claire", "Daniel"),
  female=c(T,F,T,F),
  age=c(20,25,30,35)
)
l$name[2]
l[[1]][2]


#Matrix
A=matrix(
  1:6,
  nrow=2,
  ncol=3,
  byrow = T
)
A

#DataFrame

course <- c("CIS8392","CIS8010","CIS8050","CIS8398")
num_of_students <- c(20,10,40,30)
analytics_course <- c(TRUE,FALSE,TRUE,TRUE)
df= data.frame(course,n_students=num_of_students,analytics_course)
df

df[2,]
df[c(1,3),]

name=c("Alice", "Bob", "Claire", "Daniel")
female=c(T,F,T,F)
age=c(20,25,30,35)
df=data.frame(name,female,age)
row.names(df)=c("row_1","row_2","row_3","row_4")
df
mean(df$age)
