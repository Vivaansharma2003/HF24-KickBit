**Distributed Vehicle Monitoring System Using Cameras**
Overview
This project implements a Distributed Vehicle Monitoring System that leverages cameras mounted on vehicles to monitor and analyze various aspects of driving behavior, vehicle condition, and surrounding environments. The system is designed to enhance safety, provide real-time insights, and facilitate efficient fleet management.

Features:

Real-time Video Streaming: Capture and stream live video feeds from vehicle-mounted cameras and processing them in light weight scalable YOLOv8 Models.

Surrounding Environment Monitoring: Detect obstacles, traffic signs, and road conditions using computer vision techniques.

Data Storage and Retrieval: Store video footage and analysis results in a distributed database for easy access and review using MongoDB.

Alerts and Notifications: Generate alerts for unusual driving behavior or potential hazards.
User Dashboard: Provide a web-based dashboard for users to monitor and manage their vehicles.
System Architecture
The system consists of the following components:

Cameras: High-definition cameras installed on vehicles for capturing video.
Edge Processing Units: Devices that process video feeds locally to reduce latency and bandwidth usage.
Distributed Database: A cloud-based or decentralized storage solution for storing video data and analytics.
User Interface: A web application for users to view live feeds, analyze data, and receive notifications.
