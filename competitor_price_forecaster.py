import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

class CompetitorPriceForecaster:
    """Main forecasting class with LightGBM ensemble"""

    def __init__(self, target_competitors):
        self.target_competitors = target_competitors
        self.models = {}
        self.scalers = {}
        self.feature_names = None

    def prepare_data(self, df, feature_cols, target_competitor):
        """Prepare data for a specific competitor"""
        comp_data = df[df['competitor'] == target_competitor].copy()
        X = comp_data[feature_cols].fillna(0)
        y = comp_data['final_price']
        return X, y, comp_data

    def create_lgb_model(self):
        """Create LightGBM model with optimal parameters"""
        return lgb.LGBMRegressor(
            objective='regression',
            metric='mape',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42
        )

    def create_ensemble_model(self):
        """Create ensemble model with LightGBM, Ridge, and Linear"""
        lgb_model = self.create_lgb_model()
        ridge_model = Ridge(alpha=1.0, random_state=42)
        linear_model = LinearRegression()

        ensemble = VotingRegressor([
            ('lgb', lgb_model),
            ('ridge', ridge_model),
            ('linear', linear_model)
        ], weights=[0.6, 0.3, 0.1])  # Higher weight for LightGBM

        return ensemble

    def fit(self, modeling_df, feature_cols, train_dates, val_dates):
        """Fit models for all target competitors"""

        self.feature_names = feature_cols

        for competitor in self.target_competitors:
            print(f"\n=== Training model for {competitor} ===")

            # Prepare training data
            train_data = modeling_df[modeling_df['date'].isin(train_dates)]
            X_train, y_train, _ = self.prepare_data(train_data, feature_cols, competitor)

            # Prepare validation data
            val_data = modeling_df[modeling_df['date'].isin(val_dates)]
            X_val, y_val, _ = self.prepare_data(val_data, feature_cols, competitor)

            if len(X_train) == 0 or len(X_val) == 0:
                print(f"⚠️ No data available for {competitor}")
                continue

            # Create and fit scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Create and fit ensemble model
            ensemble = self.create_ensemble_model()
            ensemble.fit(X_train_scaled, y_train)

            # Validate
            val_pred = ensemble.predict(X_val_scaled)
            val_mape = mean_absolute_percentage_error(y_val, val_pred)

            print(f"Validation MAPE for {competitor}: {val_mape:.4f}")

            # Store model and scaler
            self.models[competitor] = ensemble
            self.scalers[competitor] = scaler

        print("\n✅ All models trained successfully!")

    def predict(self, X, competitor):
        """Make predictions for a specific competitor"""
        if competitor not in self.models:
            raise ValueError(f"No model found for competitor {competitor}")

        X_scaled = self.scalers[competitor].transform(X)
        return self.models[competitor].predict(X_scaled)

    def predict_all(self, X):
        """Make predictions for all competitors"""
        predictions = {}
        for competitor in self.target_competitors:
            if competitor in self.models:
                predictions[competitor] = self.predict(X, competitor)
        return predictions