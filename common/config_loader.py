# common/config_loader.py
import os
import json
import yaml # Requires PyYAML
import logging
from google.cloud import firestore

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Handles loading configurations from environment variables, files, or Firestore.
    Prioritizes environment variables, then Firestore, then local files.
    """
    def __init__(self, gcp_project_id=None, 
                 firestore_collection="agent_configs_tradeveaver", # Default collection
                 default_config_file_path=None): # e.g., "config/defaults.yaml"
        
        self.gcp_project_id = gcp_project_id or os.environ.get("GCP_PROJECT_ID")
        self.firestore_collection_name = firestore_collection
        self.default_config_file_path = default_config_file_path
        self.db_client = None

        if self.gcp_project_id:
            try:
                self.db_client = firestore.Client(project=self.gcp_project_id)
                logger.info(f"ConfigLoader: Firestore client initialized for project '{self.gcp_project_id}'.")
            except Exception as e:
                logger.warning(f"ConfigLoader: Failed to initialize Firestore client: {e}. Firestore configs will be unavailable.", exc_info=True)
                self.db_client = None
        else:
            logger.info("ConfigLoader: GCP_PROJECT_ID not set. Firestore configs will be unavailable.")

    def get_config(self, config_name, default_value=None, expected_type=None):
        """
        Retrieves a configuration value.
        Order of precedence:
        1. Environment variable (uppercase `config_name`).
        2. Firestore document (if `config_doc_id` is provided and `config_name` is a key).
           (This method is simplified; for full doc, use `get_firestore_config_doc`)
        3. Value from default_config_file_path (if `config_name` is a key).
        4. Provided `default_value`.

        Args:
            config_name (str): The name of the configuration parameter (e.g., "API_KEY", "PROFIT_TARGET").
            default_value: The value to return if not found anywhere.
            expected_type (type, optional): If provided, attempts to cast the loaded value.
        Returns:
            The configuration value, cast to `expected_type` if specified, or `default_value`.
        """
        value = None
        source = "default"

        # 1. Check Environment Variable (convert config_name to UPPER_SNAKE_CASE for env var lookup)
        env_var_name = config_name.upper() 
        env_value = os.environ.get(env_var_name)
        if env_value is not None:
            value = env_value
            source = "environment variable"
        else:
            # (Simplified: Not directly looking up individual keys from Firestore here, use get_firestore_config_doc for that)
            # (Simplified: Not directly looking up individual keys from local file here, use _load_from_file for that)
            value = default_value # Fallback to default if not in env for this simple getter

        # Type casting
        if value is not None and expected_type is not None:
            try:
                if expected_type == bool: # Special handling for bool from string
                    if isinstance(value, str):
                        value = value.lower() in ['true', '1', 't', 'y', 'yes']
                    else:
                        value = bool(value)
                else:
                    value = expected_type(value)
            except (ValueError, TypeError) as e:
                logger.warning(f"ConfigLoader: Could not cast config '{config_name}' (value: '{value}', source: {source}) "
                               f"to type {expected_type}. Using default or uncasted. Error: {e}")
                # Revert to original default if casting fails badly, or original uncasted value
                value = default_value if source == "default" else os.environ.get(env_var_name, default_value)


        logger.debug(f"ConfigLoader: Loaded '{config_name}' = '{value}' (Type: {type(value)}) from {source}.")
        return value if value is not None else default_value


    def _load_from_file(self, filepath):
        """Loads configuration from a JSON or YAML file."""
        if not filepath or not os.path.exists(filepath):
            logger.debug(f"ConfigLoader: File not found or path not specified: {filepath}")
            return {}
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                    config_data = yaml.safe_load(f)
                elif filepath.endswith(".json"):
                    config_data = json.load(f)
                else:
                    logger.warning(f"ConfigLoader: Unsupported file type for config: {filepath}. Only JSON/YAML.")
                    return {}
            logger.info(f"ConfigLoader: Loaded configuration from file: {filepath}")
            return config_data if isinstance(config_data, dict) else {}
        except Exception as e:
            logger.error(f"ConfigLoader: Error loading configuration from file {filepath}: {e}", exc_info=True)
            return {}

    def get_firestore_config_doc(self, document_id):
        """
        Retrieves a full configuration document from Firestore.
        Args:
            document_id (str): The ID of the document in the configured Firestore collection.
        Returns:
            dict: The configuration document data, or an empty dict if not found or error.
        """
        if not self.db_client:
            logger.warning("ConfigLoader: Firestore client not available. Cannot fetch Firestore config doc.")
            return {}
        if not document_id:
            logger.error("ConfigLoader: Firestore document_id not provided.")
            return {}
            
        try:
            doc_ref = self.db_client.collection(self.firestore_collection_name).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                config_data = doc.to_dict()
                logger.info(f"ConfigLoader: Successfully loaded config document '{document_id}' from Firestore collection '{self.firestore_collection_name}'.")
                return config_data
            else:
                logger.warning(f"ConfigLoader: Config document '{document_id}' not found in Firestore collection '{self.firestore_collection_name}'.")
                return {}
        except Exception as e:
            logger.error(f"ConfigLoader: Error fetching config document '{document_id}' from Firestore: {e}", exc_info=True)
            return {}

    def get_full_config(self, firestore_doc_id=None):
        """
        Loads configuration by merging sources: File -> Firestore -> Environment Variables.
        Environment variables override values from files and Firestore for individual keys if they exist.
        Firestore document overrides file content.
        Args:
            firestore_doc_id (str, optional): Document ID in Firestore to load as base.
        Returns:
            dict: The merged configuration.
        """
        merged_config = {}

        # 1. Load from default file if specified
        if self.default_config_file_path:
            file_config = self._load_from_file(self.default_config_file_path)
            merged_config.update(file_config)
            logger.debug(f"Config after file load: {list(merged_config.keys())}")


        # 2. Load from Firestore and override/merge with file config
        if firestore_doc_id:
            fs_config = self.get_firestore_config_doc(firestore_doc_id)
            if fs_config: # Only update if fs_config is not empty
                merged_config.update(fs_config) # Firestore overrides file
                logger.debug(f"Config after Firestore load: {list(merged_config.keys())}")


        # 3. Override with Environment Variables (for keys present in merged_config or common patterns)
        # This part is tricky for a generic loader. Typically, you'd explicitly check env vars
        # for keys you expect to be overridable.
        # For simplicity, let's assume env vars are primarily for secrets or top-level settings
        # and are fetched directly using get_config("MY_SECRET_KEY_ENV_NAME").
        # A more advanced merge would iterate os.environ and map relevant vars.
        # For now, the `get_config` method already prioritizes env vars for individual lookups.
        # This `get_full_config` provides a base from files/Firestore.
        
        logger.info(f"ConfigLoader: Final merged base configuration loaded (before individual env overrides via get_config). Keys: {list(merged_config.keys())}")
        return merged_config


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Create dummy config files for testing
    if not os.path.exists("config"): os.makedirs("config")
    with open("config/defaults.json", "w") as f:
        json.dump({"DEFAULT_SETTING_FILE": "json_file_value", "LOG_LEVEL_FILE": "INFO", "TIMEOUT_MS_FILE": 5000}, f)
    with open("config/defaults.yaml", "w") as f:
        yaml.dump({"DEFAULT_SETTING_FILE": "yaml_file_value", "LOG_LEVEL_FILE": "DEBUG", "RETRY_ATTEMPTS_FILE": 3}, f)

    # --- Test with JSON file ---
    logger.info("\n--- Testing ConfigLoader with JSON default file ---")
    loader_json = ConfigLoader(default_config_file_path="config/defaults.json")
    full_conf_json = loader_json.get_full_config()
    logger.info(f"Full config from JSON file: {full_conf_json}")
    log_level = loader_json.get_config("LOG_LEVEL_FILE", default_value="WARNING", expected_type=str)
    logger.info(f"LOG_LEVEL_FILE (from JSON, env fallback): {log_level}")

    # --- Test with YAML file ---
    logger.info("\n--- Testing ConfigLoader with YAML default file ---")
    loader_yaml = ConfigLoader(default_config_file_path="config/defaults.yaml")
    full_conf_yaml = loader_yaml.get_full_config()
    logger.info(f"Full config from YAML file: {full_conf_yaml}")
    retry_attempts = loader_yaml.get_config("RETRY_ATTEMPTS_FILE", default_value=1, expected_type=int)
    logger.info(f"RETRY_ATTEMPTS_FILE (from YAML, env fallback): {retry_attempts}")

    # --- Test with Environment Variable Override ---
    logger.info("\n--- Testing Environment Variable Override ---")
    os.environ["LOG_LEVEL_FILE"] = "ERROR_FROM_ENV" # Override file value
    os.environ["NEW_ENV_CONFIG"] = "new_value_from_env"
    os.environ["TIMEOUT_MS_FILE"] = "not_an_int" # Test type casting failure
    os.environ["BOOLEAN_CONFIG_ENV"] = "true"

    log_level_env = loader_yaml.get_config("LOG_LEVEL_FILE", default_value="WARNING", expected_type=str)
    logger.info(f"LOG_LEVEL_FILE (expecting ENV override): {log_level_env}")
    assert log_level_env == "ERROR_FROM_ENV"
    
    new_conf = loader_yaml.get_config("NEW_ENV_CONFIG", default_value="default_new", expected_type=str)
    logger.info(f"NEW_ENV_CONFIG (expecting ENV value): {new_conf}")
    assert new_conf == "new_value_from_env"

    timeout_val = loader_yaml.get_config("TIMEOUT_MS_FILE", default_value=1000, expected_type=int)
    logger.info(f"TIMEOUT_MS_FILE (expecting type cast failure, then default): {timeout_val}")
    assert timeout_val == 1000 # Should use default due to cast error from "not_an_int"

    bool_val = loader_yaml.get_config("BOOLEAN_CONFIG_ENV", default_value=False, expected_type=bool)
    logger.info(f"BOOLEAN_CONFIG_ENV (expecting True): {bool_val}")
    assert bool_val is True

    # --- Test Firestore (conceptual, requires GCP setup or emulator) ---
    # logger.info("\n--- Testing Firestore Config (Conceptual) ---")
    # # Assuming GCP_PROJECT_ID is set and Firestore is accessible
    # # And a document "my_app_settings" exists in "agent_configs_tradeveaver"
    # # with {"firestore_specific_key": "value_from_firestore", "LOG_LEVEL_FILE": "WARN_FROM_FS"}
    # loader_fs = ConfigLoader(gcp_project_id=os.environ.get("GCP_PROJECT_ID"), 
    #                          firestore_collection="agent_configs_tradeveaver",
    #                          default_config_file_path="config/defaults.yaml")
    # full_conf_with_fs = loader_fs.get_full_config(firestore_doc_id="my_app_settings")
    # logger.info(f"Full config with Firestore: {full_conf_with_fs}")
    # fs_specific = loader_fs.get_config("firestore_specific_key", default_value="not_found") # This won't pick from full_conf_with_fs directly
    # logger.info(f"Firestore specific key (direct get): {fs_specific}") 
    # # To get from the merged dict:
    # if full_conf_with_fs:
    #    logger.info(f"Firestore specific key (from merged): {full_conf_with_fs.get('firestore_specific_key')}")
    #    logger.info(f"LOG_LEVEL_FILE (from merged, FS should override YAML, ENV should override FS): {full_conf_with_fs.get('LOG_LEVEL_FILE')}")
    #    # Note: get_config() itself prioritizes ENV, so it would still show ERROR_FROM_ENV for LOG_LEVEL_FILE

    # Cleanup dummy files
    os.remove("config/defaults.json")
    os.remove("config/defaults.yaml")
    os.rmdir("config")
