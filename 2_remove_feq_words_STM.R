pacman::p_load(tidyverse, quanteda, quanteda.corpora, quanteda.textstats, readxl,stm)



#################### Extract Frequency Words #########################
df <- read_excel("D:/ILDC/send_inR_find_freq_rare_words.xlsx")

df_corpus  <- corpus(df, text_field="clean_text_stopword_removed")

# pre-process and convert to dfm
df_dfm <- tokens(df_corpus,
                     remove_punct = TRUE, 
                     remove_numbers= TRUE, verbose = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  dfm() 

trim = dfm_trim(df_dfm, 
                    min_docfreq = 0.05,
                    max_docfreq = 0.90,
                    docfreq_type ="prop", 
                    verbose = TRUE)

original_terms <- featnames(df_dfm)
trimmed_terms <- featnames(trim)

removed_words <- setdiff(original_terms, trimmed_terms)

##extract removed words to be used in 2nd pre-processing script in python 
write.csv(removed_words, "D:/ILDC/removed_words.csv", row.names = FALSE, quote = FALSE)



#################### Conduct STM #########################
df_dfm <- tokens(df$clean_text_stopword_removed,
                 remove_punct = TRUE, 
                 remove_numbers= TRUE, verbose = TRUE) %>%
  dfm %>%
  dfm_wordstem() %>%
  dfm_remove(stopwords("en"))

trim = dfm_trim(df_dfm, 
                min_docfreq = 0.05,
                max_docfreq = 0.90,
                docfreq_type ="prop", 
                verbose = TRUE)
#adding metadata
trim$head <- trim$clean_text_stopword_removed
overall_dfm_non_na <-  trim[rowSums(trim) > 0,]

#converting to stm appropriate object
dfm_stm <-  quanteda::convert(overall_dfm_non_na, to = "stm")

#searching for best topic model 
many_models_search_k <- searchK(dfm_stm$documents, dfm_stm$vocab,
                                K = seq(2, 10, 1),
                                data = dfm_stm$meta,
                                init.type = "Spectral")

plot(many_models_search_k)