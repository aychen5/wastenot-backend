-- Emissions Results table
-- This table stores calculated emission results from both meal emissions and component-based calculations

CREATE TABLE IF NOT EXISTS emissions_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text,
  session_id text,
  meal_name text,  -- Name of the meal (for meal_emissions endpoint)
  leftover_g numeric,  -- Total leftover grams
  items jsonb NOT NULL,  -- Array of component results with emissions breakdown
  totals jsonb NOT NULL,  -- Aggregated totals (landfill_kgco2e, compost_kgco2e, avoided_kgco2e)
  source text,  -- Source of the calculation (e.g., 'meal_emissions', 'lovable')
  warm_version text,  -- WARM version used for calculations
  created_at timestamptz DEFAULT now()
);

-- Optional: Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_emissions_results_user_id ON emissions_results(user_id);
CREATE INDEX IF NOT EXISTS idx_emissions_results_session_id ON emissions_results(session_id);
CREATE INDEX IF NOT EXISTS idx_emissions_results_created_at ON emissions_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_emissions_results_source ON emissions_results(source);

-- Optional: Add comments for documentation
COMMENT ON TABLE emissions_results IS 'Stores calculated greenhouse gas emissions for food waste items and meals';
COMMENT ON COLUMN emissions_results.items IS 'JSONB array of component-level emissions calculations';
COMMENT ON COLUMN emissions_results.totals IS 'JSONB object with aggregated landfill_kgco2e, compost_kgco2e, and avoided_kgco2e';
COMMENT ON COLUMN emissions_results.source IS 'Identifies the calculation source: meal_emissions or lovable';
COMMENT ON COLUMN emissions_results.warm_version IS 'Version of WARM (Waste Reduction Model) used for emission factors';

