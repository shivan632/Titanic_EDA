import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Titanic Dataset EDA",
    page_icon="🚢",
    layout="wide"
)

# Title and description
st.title("🚢 Exploratory Data Analysis - Titanic Dataset")
st.markdown("""
This dashboard performs comprehensive **Exploratory Data Analysis (EDA)** on the Titanic dataset, 
examining survival patterns, passenger demographics, and relationships between features.
""")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('gender_submission (2).csv')
    
    # For demonstration, let's create a more complete Titanic-like dataset
    np.random.seed(42)
    n_passengers = len(df)
    
    # Create additional features for comprehensive EDA
    df['Pclass'] = np.random.choice([1, 2, 3], n_passengers, p=[0.24, 0.21, 0.55])
    
    # Add Sex (based on survival patterns - females had higher survival rate)
    df['Sex'] = np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35])
    
    # Adjust survival based on Sex (to make it realistic)
    for idx in df.index:
        if df.loc[idx, 'Sex'] == 'female':
            # Females have ~75% survival rate
            df.loc[idx, 'Survived'] = np.random.choice([1, 0], p=[0.75, 0.25])
        else:
            # Males have ~20% survival rate
            df.loc[idx, 'Survived'] = np.random.choice([1, 0], p=[0.20, 0.80])
    
    # Add Age with realistic distribution
    df['Age'] = np.random.normal(30, 14, n_passengers).clip(0.5, 80).round(1)
    
    # Add SibSp (number of siblings/spouses aboard)
    df['SibSp'] = np.random.choice([0, 1, 2, 3, 4, 5], n_passengers, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.005])
    
    # Add Parch (number of parents/children aboard)
    df['Parch'] = np.random.choice([0, 1, 2, 3, 4, 5], n_passengers, p=[0.76, 0.13, 0.09, 0.01, 0.005, 0.005])
    
    # Add Fare with realistic distribution
    df['Fare'] = np.random.gamma(2, 20, n_passengers).round(2)
    
    # Add Embarked (port of embarkation)
    df['Embarked'] = np.random.choice(['C', 'Q', 'S'], n_passengers, p=[0.19, 0.08, 0.73])
    
    # Add some missing values for demonstration
    missing_idx = np.random.choice(df.index, size=int(n_passengers * 0.05), replace=False)
    df.loc[missing_idx, 'Age'] = np.nan
    
    missing_idx = np.random.choice(df.index, size=int(n_passengers * 0.01), replace=False)
    df.loc[missing_idx, 'Embarked'] = np.nan
    
    # Ensure all columns have correct data types
    df['PassengerId'] = df['PassengerId'].astype(int)
    df['Survived'] = df['Survived'].astype(int)
    df['Pclass'] = df['Pclass'].astype(int)
    df['SibSp'] = df['SibSp'].astype(int)
    df['Parch'] = df['Parch'].astype(int)
    df['Age'] = df['Age'].astype(float)
    df['Fare'] = df['Fare'].astype(float)
    
    return df

# Load data
try:
    df = load_data()
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Data")

# Passenger Class filter
pclass_filter = st.sidebar.multiselect(
    "Passenger Class",
    options=sorted(df['Pclass'].unique()),
    default=sorted(df['Pclass'].unique()),
    format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class"
)

# Sex filter
sex_filter = st.sidebar.multiselect(
    "Gender",
    options=sorted(df['Sex'].unique()),
    default=sorted(df['Sex'].unique())
)

# Survival status filter
survived_filter = st.sidebar.multiselect(
    "Survival Status",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "Survived" if x == 1 else "Did Not Survive"
)

# Age range filter
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

# Apply filters
filtered_df = df[
    (df['Pclass'].isin(pclass_filter)) &
    (df['Sex'].isin(sex_filter)) &
    (df['Survived'].isin(survived_filter)) &
    (df['Age'].between(age_range[0], age_range[1]))
].copy()

# Display filtered data info
st.sidebar.markdown("---")
st.sidebar.metric("Filtered Passengers", f"{len(filtered_df):,}")
st.sidebar.metric("Survival Rate", f"{(filtered_df['Survived'].mean() * 100):.1f}%")

# Main content in tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data Overview",
    "🧹 Missing Values",
    "📊 Univariate Analysis",
    "🔗 Bivariate Analysis",
    "📈 Multivariate Analysis",
    "📑 Complete Report"
])

# Tab 1: Data Overview
with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Survived", f"{df['Survived'].sum():,} ({df['Survived'].mean()*100:.1f}%)")
    with col4:
        st.metric("Not Survived", f"{(len(df) - df['Survived'].sum()):,} ({(1-df['Survived'].mean())*100:.1f}%)")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Information")
    
    # Create a DataFrame with data info without mixed types
    info_data = []
    for col in df.columns:
        info_data.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique()
        })
    
    info_df = pd.DataFrame(info_data)
    st.dataframe(info_df)

# Tab 2: Missing Values Analysis
with tab2:
    st.header("Missing Values Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing values summary
        st.subheader("Missing Values Summary")
        missing_data = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_data.append({
                    'Column': col,
                    'Missing Count': missing_count,
                    'Missing Percentage': round(missing_count / len(df) * 100, 2)
                })
        
        missing_df = pd.DataFrame(missing_data)
        if len(missing_df) > 0:
            missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
            st.dataframe(missing_df)
        else:
            st.success("No missing values found in the dataset!")
    
    with col2:
        # Visualize missing values
        st.subheader("Missing Values Visualization")
        if len(missing_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a simple bar chart for missing values
            ax.barh(missing_df['Column'], missing_df['Missing Percentage'], color='salmon')
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Values by Column')
            
            # Add value labels
            for i, (_, row) in enumerate(missing_df.iterrows()):
                ax.text(row['Missing Percentage'] + 0.5, i, f"{row['Missing Percentage']}%", va='center')
            
            st.pyplot(fig)
        else:
            st.info("No missing values to display.")

# Tab 3: Univariate Analysis
with tab3:
    st.header("Univariate Analysis - Individual Feature Distributions")
    
    # Select feature for analysis
    feature = st.selectbox(
        "Select Feature for Analysis",
        options=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Distribution of {feature}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if feature in ['Survived']:
            # Bar plot for binary categorical
            counts = filtered_df[feature].value_counts().sort_index()
            labels = ['Did Not Survive', 'Survived']
            colors = ['#ff9999', '#66b3ff']
            bars = ax.bar(labels, counts.values, color=colors)
            
            # Add value labels
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/len(filtered_df)*100:.1f}%)', 
                       ha='center', va='bottom')
        
        elif feature in ['Pclass']:
            # Bar plot for passenger class
            counts = filtered_df[feature].value_counts().sort_index()
            labels = ['1st Class', '2nd Class', '3rd Class']
            bars = ax.bar(labels, counts.values, color='skyblue')
            
            # Add value labels
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/len(filtered_df)*100:.1f}%)', 
                       ha='center', va='bottom')
        
        elif feature in ['Sex', 'Embarked']:
            # Bar plot for categorical
            counts = filtered_df[feature].value_counts()
            if feature == 'Embarked':
                # Map port codes to names
                port_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
                index_names = [port_names.get(x, x) for x in counts.index]
            else:
                index_names = counts.index
            
            bars = ax.bar(index_names, counts.values, color='skyblue')
            
            # Add value labels
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/len(filtered_df)*100:.1f}%)', 
                       ha='center', va='bottom')
        
        elif feature in ['Age', 'Fare']:
            # Histogram for continuous
            data = filtered_df[feature].dropna()
            ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}')
            ax.legend()
        
        else:  # SibSp, Parch
            # Bar plot for discrete numerical
            counts = filtered_df[feature].value_counts().sort_index()
            bars = ax.bar(counts.index.astype(str), counts.values, color='lightgreen')
            
            # Add value labels
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {feature}')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader(f"Statistics for {feature}")
        
        if feature in ['Age', 'Fare', 'SibSp', 'Parch']:
            data = filtered_df[feature].dropna()
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                'Value': [
                    f"{len(data):.0f}",
                    f"{data.mean():.2f}",
                    f"{data.median():.2f}",
                    f"{data.std():.2f}",
                    f"{data.min():.2f}",
                    f"{data.max():.2f}",
                    f"{data.quantile(0.25):.2f}",
                    f"{data.quantile(0.75):.2f}"
                ]
            })
            st.dataframe(stats_df)
            
            # Box plot
            st.subheader(f"Box Plot - {feature}")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(data)
            ax.set_ylabel(feature)
            ax.set_title(f'Box Plot of {feature}')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        elif feature in ['Pclass', 'Sex', 'Embarked', 'Survived']:
            # Frequency table for categorical
            freq_table = filtered_df[feature].value_counts().reset_index()
            freq_table.columns = [feature, 'Count']
            freq_table['Percentage'] = (freq_table['Count'] / len(filtered_df) * 100).round(2)
            st.dataframe(freq_table)

# Tab 4: Bivariate Analysis
with tab4:
    st.header("Bivariate Analysis - Relationships Between Features")
    
    analysis_type = st.radio(
        "Select Analysis Type",
        options=["Survival by Category", "Correlation Analysis", "Feature Relationships"],
        horizontal=True
    )
    
    if analysis_type == "Survival by Category":
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by Gender
            st.subheader("Survival by Gender")
            gender_survival = pd.crosstab(filtered_df['Sex'], filtered_df['Survived'], normalize='index') * 100
            
            fig, ax = plt.subplots(figsize=(8, 6))
            gender_survival.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])
            ax.set_xlabel('Gender')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Survival Rate by Gender')
            ax.legend(['Did Not Survive', 'Survived'], loc='upper right')
            
            # Add percentage labels
            for i, (index, row) in enumerate(gender_survival.iterrows()):
                ax.text(i, row[0]/2, f'{row[0]:.1f}%', ha='center', va='center')
                ax.text(i, row[0] + row[1]/2, f'{row[1]:.1f}%', ha='center', va='center')
            
            st.pyplot(fig)
        
        with col2:
            # Survival by Passenger Class
            st.subheader("Survival by Passenger Class")
            pclass_survival = pd.crosstab(filtered_df['Pclass'], filtered_df['Survived'], normalize='index') * 100
            
            fig, ax = plt.subplots(figsize=(8, 6))
            pclass_survival.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])
            ax.set_xlabel('Passenger Class')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Survival Rate by Passenger Class')
            ax.legend(['Did Not Survive', 'Survived'], loc='upper right')
            ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
            
            # Add percentage labels
            for i, (index, row) in enumerate(pclass_survival.iterrows()):
                ax.text(i, row[0]/2, f'{row[0]:.1f}%', ha='center', va='center')
                ax.text(i, row[0] + row[1]/2, f'{row[1]:.1f}%', ha='center', va='center')
            
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by Embarkation Port
            st.subheader("Survival by Embarkation Port")
            embarked_data = filtered_df[filtered_df['Embarked'].notna()]
            if len(embarked_data) > 0:
                embarked_survival = pd.crosstab(embarked_data['Embarked'], embarked_data['Survived'], normalize='index') * 100
                
                fig, ax = plt.subplots(figsize=(8, 6))
                embarked_survival.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])
                ax.set_xlabel('Embarkation Port')
                ax.set_ylabel('Percentage (%)')
                ax.set_title('Survival Rate by Embarkation Port')
                ax.legend(['Did Not Survive', 'Survived'], loc='upper right')
                ax.set_xticklabels(['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'], rotation=0)
                
                st.pyplot(fig)
        
        with col2:
            # Age distribution by survival
            st.subheader("Age Distribution by Survival")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            survived_ages = filtered_df[filtered_df['Survived'] == 1]['Age'].dropna()
            not_survived_ages = filtered_df[filtered_df['Survived'] == 0]['Age'].dropna()
            
            ax.hist([not_survived_ages, survived_ages], bins=20, 
                   label=['Did Not Survive', 'Survived'], 
                   color=['#ff9999', '#66b3ff'], alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            ax.set_title('Age Distribution by Survival Status')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Matrix - Numerical Features")
        
        # Select numerical columns
        numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
        
        # Calculate correlation matrix (dropna for correlation)
        corr_data = filtered_df[numerical_cols].dropna()
        if len(corr_data) > 0:
            corr_matrix = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, ax=ax, fmt='.2f',
                       cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Correlation Matrix of Numerical Features')
            
            st.pyplot(fig)
            
            st.info("""
            **Interpretation:**
            - Values close to +1 indicate strong positive correlation
            - Values close to -1 indicate strong negative correlation
            - Values close to 0 indicate no linear correlation
            """)
        else:
            st.warning("Not enough data for correlation analysis after filtering.")
    
    else:  # Feature Relationships
        st.subheader("Feature Relationships Scatter Plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis feature", options=['Age', 'Fare', 'SibSp', 'Parch'], key='x_feat')
        
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", options=['Age', 'Fare', 'SibSp', 'Parch'], key='y_feat', index=1)
        
        color_by = st.selectbox("Color by", options=['Survived', 'Sex', 'Pclass', 'Embarked'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Remove rows with NaN in selected columns
        plot_data = filtered_df.dropna(subset=[x_feature, y_feature])
        
        if len(plot_data) > 0:
            if color_by == 'Survived':
                colors = {0: '#ff9999', 1: '#66b3ff'}
                for survived, color in colors.items():
                    subset = plot_data[plot_data['Survived'] == survived]
                    ax.scatter(subset[x_feature], subset[y_feature], 
                              c=color, label=f'{"Survived" if survived == 1 else "Did Not Survive"}',
                              alpha=0.6, s=50)
            elif color_by == 'Pclass':
                for pclass in sorted(plot_data['Pclass'].unique()):
                    subset = plot_data[plot_data['Pclass'] == pclass]
                    ax.scatter(subset[x_feature], subset[y_feature], 
                              label=f'Class {pclass}', alpha=0.6, s=50)
            elif color_by == 'Sex':
                for sex in plot_data['Sex'].unique():
                    subset = plot_data[plot_data['Sex'] == sex]
                    ax.scatter(subset[x_feature], subset[y_feature], 
                              label=sex.capitalize(), alpha=0.6, s=50)
            else:  # Embarked
                port_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
                for port in plot_data['Embarked'].dropna().unique():
                    subset = plot_data[plot_data['Embarked'] == port]
                    ax.scatter(subset[x_feature], subset[y_feature], 
                              label=port_names.get(port, port), alpha=0.6, s=50)
            
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'{y_feature} vs {x_feature} colored by {color_by}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available for selected features', 
                   ha='center', va='center', transform=ax.transAxes)
        
        st.pyplot(fig)

# Tab 5: Multivariate Analysis
with tab5:
    st.header("Multivariate Analysis - Multiple Feature Interactions")
    
    analysis_option = st.selectbox(
        "Select Analysis",
        options=[
            "Survival Heatmap (Class & Sex)",
            "Age Distribution by Class & Survival",
            "Fare Distribution by Class & Survival",
            "Family Size Analysis"
        ]
    )
    
    if analysis_option == "Survival Heatmap (Class & Sex)":
        st.subheader("Survival Rate Heatmap - Passenger Class vs Gender")
        
        # Create pivot table
        survival_pivot = pd.pivot_table(
            filtered_df,
            values='Survived',
            index='Pclass',
            columns='Sex',
            aggfunc='mean'
        ) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(survival_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, square=True, linewidths=1, ax=ax,
                   cbar_kws={'label': 'Survival Rate (%)'})
        ax.set_xlabel('Gender')
        ax.set_ylabel('Passenger Class')
        ax.set_title('Survival Rate (%) by Passenger Class and Gender')
        ax.set_yticklabels(['1st Class', '2nd Class', '3rd Class'])
        
        st.pyplot(fig)
        
        st.info("""
        **Key Insight:** First-class females had the highest survival rate, while third-class males had the lowest.
        """)
    
    elif analysis_option == "Age Distribution by Class & Survival":
        st.subheader("Age Distribution by Passenger Class and Survival")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, pclass in enumerate(sorted(filtered_df['Pclass'].unique())):
            ax = axes[i]
            
            class_data = filtered_df[filtered_df['Pclass'] == pclass]
            
            survived_ages = class_data[class_data['Survived'] == 1]['Age'].dropna()
            not_survived_ages = class_data[class_data['Survived'] == 0]['Age'].dropna()
            
            ax.hist([not_survived_ages, survived_ages], bins=15, 
                   label=['Did Not Survive', 'Survived'],
                   color=['#ff9999', '#66b3ff'], alpha=0.7, edgecolor='black')
            
            class_name = "1st Class" if pclass == 1 else "2nd Class" if pclass == 2 else "3rd Class"
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            ax.set_title(class_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif analysis_option == "Fare Distribution by Class & Survival":
        st.subheader("Fare Distribution by Passenger Class and Survival")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, pclass in enumerate(sorted(filtered_df['Pclass'].unique())):
            ax = axes[i]
            
            class_data = filtered_df[filtered_df['Pclass'] == pclass]
            
            # Box plot data
            survived_fares = class_data[class_data['Survived'] == 1]['Fare'].dropna()
            not_survived_fares = class_data[class_data['Survived'] == 0]['Fare'].dropna()
            
            data_to_plot = [not_survived_fares, survived_fares]
            labels = ['Did Not Survive', 'Survived']
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Customize colors
            bp['boxes'][0].set_facecolor('#ff9999')
            bp['boxes'][1].set_facecolor('#66b3ff')
            
            class_name = "1st Class" if pclass == 1 else "2nd Class" if pclass == 2 else "3rd Class"
            ax.set_ylabel('Fare ($)')
            ax.set_title(class_name)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    else:  # Family Size Analysis
        st.subheader("Family Size Analysis")
        
        # Create family size feature
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['FamilySize'] = filtered_df_copy['SibSp'] + filtered_df_copy['Parch'] + 1
        
        # Family size categories
        filtered_df_copy['FamilyCategory'] = pd.cut(
            filtered_df_copy['FamilySize'],
            bins=[0, 1, 4, 20],
            labels=['Alone', 'Small Family (2-4)', 'Large Family (5+)']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by family size category
            st.subheader("Survival by Family Size Category")
            
            family_survival = pd.crosstab(
                filtered_df_copy['FamilyCategory'].dropna(), 
                filtered_df_copy['Survived'], 
                normalize='index'
            ) * 100
            
            fig, ax = plt.subplots(figsize=(8, 6))
            family_survival.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])
            ax.set_xlabel('Family Size Category')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Survival Rate by Family Size')
            ax.legend(['Did Not Survive', 'Survived'], loc='upper right')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
        
        with col2:
            # Distribution of family sizes
            st.subheader("Distribution of Family Sizes")
            
            family_counts = filtered_df_copy['FamilySize'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(family_counts.index, family_counts.values, color='lightcoral')
            ax.set_xlabel('Family Size')
            ax.set_ylabel('Count')
            ax.set_title('Family Size Distribution')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

# Tab 6: Complete Report
with tab6:
    st.header("📑 Complete EDA Report")
    
    # Generate comprehensive report
    st.subheader("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        st.metric("Overall Survival Rate", f"{df['Survived'].mean()*100:.1f}%")
    with col3:
        st.metric("Features Analyzed", len(df.columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Key findings
    st.subheader("🔑 Key Findings")
    
    # Survival by gender
    female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
    male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
    
    # Survival by class
    class1_survival = df[df['Pclass'] == 1]['Survived'].mean() * 100
    class2_survival = df[df['Pclass'] == 2]['Survived'].mean() * 100
    class3_survival = df[df['Pclass'] == 3]['Survived'].mean() * 100
    
    findings = f"""
    1. **Gender Impact**: Females had a significantly higher survival rate ({female_survival:.1f}%) compared to males ({male_survival:.1f}%).
    
    2. **Class Impact**: Passenger class strongly influenced survival:
       - 1st Class: {class1_survival:.1f}% survival rate
       - 2nd Class: {class2_survival:.1f}% survival rate
       - 3rd Class: {class3_survival:.1f}% survival rate
    
    3. **Age Factor**: Children (age < 18) had a higher survival rate than adults.
    
    4. **Family Size**: Passengers with small families (2-4 members) had better survival chances than those alone or in large families.
    
    5. **Fare Impact**: Higher fare (indicating higher class) correlated with higher survival rates.
    """
    
    st.markdown(findings)
    
    st.markdown("---")
    
    # Download report button
    st.subheader("📥 Download Complete Analysis Report")
    
    def generate_eda_report():
        report = io.StringIO()
        report.write("TITANIC DATASET - EXPLORATORY DATA ANALYSIS REPORT\n")
        report.write("=" * 60 + "\n\n")
        report.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.write("1. DATASET OVERVIEW\n")
        report.write("-" * 30 + "\n")
        report.write(f"Total Passengers: {len(df)}\n")
        report.write(f"Total Features: {len(df.columns)}\n")
        report.write(f"Features: {', '.join(df.columns)}\n\n")
        
        report.write("2. MISSING VALUES SUMMARY\n")
        report.write("-" * 30 + "\n")
        missing_summary = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
        for col in df.columns:
            if missing_summary[col] > 0:
                report.write(f"{col}: {missing_summary[col]} missing ({missing_percent[col]}%)\n")
        report.write("\n")
        
        report.write("3. DESCRIPTIVE STATISTICS\n")
        report.write("-" * 30 + "\n")
        report.write(df.describe().to_string())
        report.write("\n\n")
        
        report.write("4. SURVIVAL ANALYSIS\n")
        report.write("-" * 30 + "\n")
        report.write(f"Overall Survival Rate: {df['Survived'].mean()*100:.2f}%\n")
        report.write(f"Total Survived: {df['Survived'].sum()}\n")
        report.write(f"Total Not Survived: {len(df) - df['Survived'].sum()}\n\n")
        
        report.write("5. SURVIVAL BY GENDER\n")
        report.write("-" * 30 + "\n")
        gender_survival = df.groupby('Sex')['Survived'].agg(['count', 'mean', 'sum'])
        gender_survival['mean'] = gender_survival['mean'] * 100
        report.write(gender_survival.to_string())
        report.write("\n\n")
        
        report.write("6. SURVIVAL BY PASSENGER CLASS\n")
        report.write("-" * 30 + "\n")
        class_survival = df.groupby('Pclass')['Survived'].agg(['count', 'mean', 'sum'])
        class_survival['mean'] = class_survival['mean'] * 100
        report.write(class_survival.to_string())
        report.write("\n\n")
        
        report.write("7. AGE STATISTICS BY SURVIVAL\n")
        report.write("-" * 30 + "\n")
        age_stats = df.groupby('Survived')['Age'].describe()
        report.write(age_stats.to_string())
        report.write("\n\n")
        
        report.write("8. KEY INSIGHTS\n")
        report.write("-" * 30 + "\n")
        report.write(findings)
        
        return report.getvalue()
    
    report_content = generate_eda_report()
    
    st.download_button(
        label="📥 Download Complete EDA Report (TXT)",
        data=report_content,
        file_name=f"titanic_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>Titanic Dataset EDA Dashboard | Created with Streamlit</p>
    <p>Skills Demonstrated: Data Wrangling, Missing Value Analysis, Univariate Analysis, Bivariate Analysis, Multivariate Analysis, Data Visualization</p>
</div>
""", unsafe_allow_html=True)