# Stocks by news 
# Should build model on some of the data and check for predictions afterwards

# Will be doing text analysis, using textmineR
library(tm)
library(SnowballC)


# Load the data, doing it separately so that I can do further analysis on the DJIA change
redditNews = read.csv("8YearsOfStocks\\Combined_News_DJIA.csv", strip.white = TRUE, as.is = TRUE, header = TRUE, stringsAsFactors = FALSE)
head(redditNews)


# Try to bind all of the day's titles into one field
redditNews$All = paste0(redditNews$Top1, "  ,", redditNews$Top2, "  ,", redditNews$Top3, "  ,", redditNews$Top4, "  ,",
                        redditNews$Top5, "  ,", redditNews$Top6, "  ,",
                        redditNews$Top7, "  ,", redditNews$Top8, "  ,", redditNews$Top9, "  ,", redditNews$Top10, "  ,",
                        redditNews$Top11, "  ,", redditNews$Top12, "  ,", 
                        redditNews$Top13, "  ,", redditNews$Top14, "  ,", redditNews$Top15, "  ,", redditNews$Top16, "  ,",
                        redditNews$Top17, "  ,", redditNews$Top18, "  ,",
                        redditNews$Top19, "  ,", redditNews$Top20, "  ,", redditNews$Top21, "  ,", redditNews$Top22, "  ,",
                        redditNews$Top23, "  ,", redditNews$Top24, "  ,",
                        redditNews$Top25)
redditNews$All[1]
names(redditNews)
# Corpus of all info, documents of each date's 25 posts

myReader = readTabular(mapping = list(content = names(redditNews)[3:27], id = "Date"))
myReader

docs  = VCorpus(DataframeSource(redditNews), readerControl = list(reader = myReader))

inspect(docs)[1:5]

writeLines(as.character(docs[[1]])) # Index respresents which item of corpus being used when double selected

# Preprocessing

for(j in seq(docs))   
{   
  docs[[j]] = gsub(",b", " ", docs[[j]])
  docs[[j]] = gsub("/", " ", docs[[j]])   
  docs[[j]] = gsub("@", " ", docs[[j]])   
  docs[[j]] = gsub("\\|", " ", docs[[j]])   
} 

writeLines(as.character(docs[[1]]))

docs = tm_map(docs, tolower)
inspect(docs[3])

docs = tm_map(docs, removeWords, stopwords("english"))

# These words are muddying up analysis while not adding anything of value
docs = tm_map(docs, removeWords, c("new", "will", "says"))

docs = tm_map(docs, stemDocument) #ing, es, other common endings

docs = tm_map(docs, stripWhitespace)

docs = tm_map(docs, PlainTextDocument)
inspect(docs[1])


# Staging the data
newsDTM = DocumentTermMatrix(docs)
newsDTM

# 1989 rows means each day's worth of data has been stored as a doc
# Number of columns count  unique words used, minus punctuation and others

dim(newsDTM)
inspect(newsDTM[1:7, 1:7])

# Transpose of the data

transDTM = TermDocumentMatrix(docs)
transDTM
inspect(transDTM[1:10, 1:10])

# Exploration
freq = colSums(as.matrix(newsDTM))
length(freq) # This is equal to the number of columns as it is a sum of each column

ord = order(freq)

# I will do analysis on original DTM and other variations of sparsity
nineFiveSparse = removeSparseTerms(newsDTM, .95) # This gets rid of the terms that occur in fewer than 5% of docs
NCOL(nineFiveSparse)

eightZeroSparse = removeSparseTerms(newsDTM, 0.80) # This removes the terms that occur in fewer than 20% of the docs
NCOL(eightZeroSparse)

# 12 words after removing stop words and fillers
sixZeroSparse   = removeSparseTerms(newsDTM, 0.60) # This removes the terms that occur in fewer than 40% of the docs
head(inspect(sixZeroSparse))

# 3 words
fiveZeroSparse = removeSparseTerms(newsDTM, 0.50) # This removes the terms that occur in fewer than 50% of the docs
head(inspect(fiveZeroSparse))


# Looking at the least frequent words
head(freq[ord])
# Looking at the most frequent words
tail(freq[ord])

# The following are to be read as:
# Row 1- Frequency of a term in corpus
# Row 2- Number of terms that occur exactlythat many times

# Least frequent words, occuring 1-20 times
head(table(freq), 20)
# Most frequent words
tail(table(freq), 20)

# Clustering to better understand topic models
library(cluster)
distance = dist(t(sixZeroSparse), method = "euclidean")
fit = hclust(d = distance, method = "ward.D")
fit

plot(fit, hang = -1)

# Add rectangles for readability
plot.new()
plot(fit, hang = -1)
groups = cutree(fit, k = 5)
rect.hclust(fit, k = 5, border = "red")


# fit some LDA models and select the best number of topics
rownames(newsDTM) = redditNews$Date
rownames(newsDTM)

rownames(nineFiveSparse) = redditNews$Date
rownames(nineFiveSparse)


library(topicmodels)
burnin = 4000
iter = 2000
thin = 500
seed = list(2003)
nstart = 1
best = TRUE

# The code will allow me to make topic number selection
# minimum or Arun and Juan, max of the other 2 methods indicate optimal topic number
# Takes roughly an hour to run on desktop
# Would be interested to see results on eightZeroSparse
library(ldatuning)
result = FindTopicsNumber(
  nineFiveSparse,
  topics = seq(from = 10, to = 150, by = 10),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result)

# The plot will make for a great way of explaining model selection in presentation

# k = optimal number of topics suggested by above
# Probably to be done as an average of the 4 methods suggested topic quantities

k = 130

# Terminates after about 10 minutes on k = 5
LDAResults= LDA(nineFiveSparse,
                k = k,
                method = "Gibbs",
                control = list(nstart = 1,
                               seed = seed,
                               best = best,
                               burnin = burnin,
                               iter = iter,
                               thin = thin))

# This matrix holds the topic corresponding to each document
LDATopics = as.matrix(topics(LDAResults))
head(LDATopics)

# This produces the top 6 terms in each topic
LDATerms = as.matrix(terms(LDAResults, 6))
names(LDATerms)
head(LDATerms)

# Probability of each topic assignment
topicProbabilities = as.data.frame(LDAResults@gamma)
names(topicProbabilities)
head(topicProbabilities)


# Time to try and create a linear model
# I will do a few approaches
# A Model each topic as continuous, giving the values of xi to be the probability of that topic during that day (document)
# B???????? Model where each variable is an indicator, 1 if topic is the best fit and 0 otherwise

# I will then run through different numbers of topics to see if I can generate more accurate predictions
DJIAData = read.csv("8YearsOfStocks\\DJIA_table.csv", header = TRUE, as.is = TRUE, strip.white = TRUE)
head(DJIAData)

# Needs to be reversed
sort(DJIAData$Date, decreasing = FALSE)
head(DJIAData)
fixed = DJIAData[order(DJIAData$Date, decreasing = FALSE),]
head(fixed)

# Add the daily change column
fixed$Change = fixed$Close - fixed$Open
head(fixed$Change)

# Adjust incorrect label assignment. Error found in check between fixed change and label
head(redditNews$Label)
redditNews$Label[1] = 1
head(redditNews$Label)

# A) -
probDF = as.data.frame(topicProbabilities, redditNews$Date)
probDF$DJIA = redditNews$Label
probDF$DJIAChange = fixed$Change
head(probDF)
names(probDF)

# Run AIC and BIC on this
# My attempt at making max size model
asProbabilities = lm(data = subset(probDF, select = -DJIA), DJIAChange ~ . - 1)
summary(asProbabilities)


# Model selection for GLM approaches
asProbabilitiesGLM0 = glm(data = probDF, DJIA ~ 1, family = "binomial")
summary(asProbabilitiesGLM0)

asProbabilitiesGLMFull = glm(data = subset(probDF, select = -DJIAChange), DJIA ~ . - 1, family = "binomial")
summary(asProbabilitiesGLMFull)

# Forwards model selection resulted in empty model
AICprobabilities = step(asProbabilitiesGLM0, scope = list(lower = ~1, upper = ~ . - 1),
                        direction = "forward", data = subset(probDF, select = -DJIAChange))

# Backwards model selection resulted in full model
AICprobabilitiesBackwards = step(asProbabilitiesGLMFull, #scope = list(lower = ~1, upper = ~ V1 + V2 + V3 + V4 + V5),
                                 direction = "backward", data = probDF)

# Results of backwards selection- Step:  AIC=2723.81
# DJIA ~ V3 + V8 + V32 + V38 + V39 + V47 + V60 + V64 + V73 + V82 + 
#   V88 + V90 + V104 + V109 + V110 + V121 + V125 + V126 + V128 - 1


# Chose a midpoint for model and ran it for both directions to see which model it naturally moved to. Ended up with full model
AICprobabilitiesBoth = step(asProbabilitiesGLMFull,
                            direction = "both", data = probDF)

AICprobabilitiesBothBIC = step(asProbabilitiesGLMFull,
                            direction = "both", data = probDF, k = log(nrow(newsDTM)))
# Step:  AIC=2750.73
# DJIA ~ V126 - 1


asProbabilitiesGLM = glm(data = probDF, DJIA ~ V3 + V8 + V32 + V38 + V39 + V47 + V60 + V64 + V73 + V82 + 
                                               V88 + V90 + V104 + V109 + V110 + V121 + V125 + V126 + V128- 1, 
                         family = "binomial")
summary(asProbabilitiesGLM)

# B
Topic = as.character(length(redditNews$All))


# Setting the topic value to 0 or 1, dependent on if it was the most probable for a given document
(LDATopics[,1])
head(LDATopics[[1]])
for( k in 1:130){
  Topic[LDATopics[,1] == k] = paste0("Topic ", k, collapse = "")
  #print(k)
}

Topic
head(Topic)
length(Topic)

asCharactersDF = as.data.frame(Topic)
head(asCharactersDF)
asCharactersDF$DJIA = probDF$DJIA
asCharactersDF$Change = probDF$DJIAChange
head(asCharactersDF)

# Very little chance of there being a relationship between topics and amount of change
asIndicatorsLM = lm(data = asCharactersDF, Change ~ as.factor(Topic) - 1)
summary(asIndicatorsLM)

# Looks like there is a good chance topics are related to direction of change
# Cant do model selection on simple linear regression

asIndicatorsGLM = glm(data = asCharactersDF, DJIA ~ factor(Topic) - 1, family = "binomial")
summary(asIndicatorsGLM)

