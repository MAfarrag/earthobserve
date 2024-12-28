"""Google earth engine main script."""
import base64
import json
import ee


class GEE:
    """GEE."""

    def __init__(self, service_account: str, service_key_path: str):
        """Initialize.

        Parameters
        ----------
        service_account: [str]
                service account name
        service_key_path: [str]
                path to the service account json file
        Returns
        -------
        None
        """
        self.initialize(service_account, service_key_path)


    @staticmethod
    def initialize(service_account: str, service_key: str):
        """Initialize.

            Initialize authenticate and initializes the connection to google earth engine with a service accont file
            content or path

        Parameters
        ----------
        service_account: [str]
                service account name
        service_key: [str]
                path to the service account json file or the content of the service account

        Returns
        -------
        None
        """
        try:
            credentials = ee.ServiceAccountCredentials(service_account, service_key)
        except ValueError:
            credentials = ee.ServiceAccountCredentials(
                service_account, key_data=service_key
            )
        ee.Initialize(credentials=credentials)
