def feature_engineer(df):
    df = df.copy()

    df['promotions_per_year'] = df['Number of Promotions'] / (df['Years at Company'] + 1) 
    df['tenure_ratio']        = df['Years at Company'] / (df['Company Tenure'] + 1)       
    df['age_at_joining']      = df['Age'] - df['Years at Company']                        
    df['income_per_dependent']= df['Monthly Income'] / (df['Number of Dependents'] + 1)   
    df['effective_distance'] = df['Distance from Home'] * (df['Remote Work'] == 'No')

    return df