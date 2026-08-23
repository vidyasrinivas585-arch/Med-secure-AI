/**
 * MedSecure AI — pharmacy-locator.js
 * Geolocation, Google Places pharmacy search, and interactive map.
 */

class PharmacyLocator {
  constructor(options = {}) {
    this.reportId = options.reportId || "";
    this.radiusKm = parseFloat(options.radiusKm) || 5;
    this.lang = options.lang || "en";
    this.showPharmacy = options.showPharmacy === true || options.showPharmacy === "true";
    this.googleMapsKey = options.googleMapsKey || "";
    this.userLat = null;
    this.userLng = null;
    this.pharmacies = [];
    this.map = null;
    this.markers = [];
    this.infoWindow = null;

    this.listEl = document.getElementById("pharmacyList");
    this.mapEl = document.getElementById("pharmacyMap");
    this.statusEl = document.getElementById("pharmacyStatus");
    this.manualSearchEl = document.getElementById("manualLocationInput");
  }

  init() {
    if (!this.showPharmacy) return;

    applyI18n(this.lang);

    const locateBtn = document.getElementById("btnUseLocation");
    const searchBtn = document.getElementById("btnSearchLocation");

    if (locateBtn) {
      locateBtn.addEventListener("click", () => this.requestLocation());
    }
    if (searchBtn) {
      searchBtn.addEventListener("click", () => this.manualSearch());
    }
    if (this.manualSearchEl) {
      this.manualSearchEl.addEventListener("keydown", e => {
        if (e.key === "Enter") this.manualSearch();
      });
    }

    // Auto-request location on unsafe predictions
    this.requestLocation();
  }

  setStatus(msg, type = "info") {
    if (!this.statusEl) return;
    const colours = { info: "text-muted", danger: "text-danger", success: "text-success", warning: "text-warning" };
    this.statusEl.className = `small ${colours[type] || colours.info}`;
    this.statusEl.textContent = msg;
  }

  requestLocation() {
    if (!navigator.geolocation) {
      this.setStatus(t("location_denied", this.lang), "warning");
      return;
    }

    this.setStatus(t("searching", this.lang));
    navigator.geolocation.getCurrentPosition(
      pos => {
        this.userLat = pos.coords.latitude;
        this.userLng = pos.coords.longitude;
        this.fetchPharmacies(this.userLat, this.userLng);
      },
      () => {
        this.setStatus(t("location_denied", this.lang), "warning");
      },
      { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 }
    );
  }

  manualSearch() {
    const query = this.manualSearchEl?.value?.trim();
    if (!query) return;
    this.setStatus(t("searching", this.lang));
    this.fetchPharmacies(null, null, query);
  }

  async fetchPharmacies(lat, lng, query) {
    try {
      let url = `/api/pharmacies?radius=${this.radiusKm}&report_id=${encodeURIComponent(this.reportId)}`;
      if (lat != null && lng != null) {
        url += `&lat=${lat}&lng=${lng}`;
      } else if (query) {
        url += `&query=${encodeURIComponent(query)}`;
      }

      const res = await fetch(url);
      const data = await res.json();

      if (!data.success) {
        this.setStatus(data.error || t("api_error", this.lang), "danger");
        this.renderList([]);
        return;
      }

      this.pharmacies = data.pharmacies || [];

      if (data.geocoded) {
        this.userLat = data.geocoded.lat;
        this.userLng = data.geocoded.lng;
      } else if (data.origin) {
        this.userLat = data.origin.lat;
        this.userLng = data.origin.lng;
      }

      if (this.pharmacies.length === 0) {
        this.setStatus(t("no_pharmacies", this.lang), "warning");
      } else {
        this.setStatus(`${this.pharmacies.length} ${t("nearby_pharmacies", this.lang)}`);
      }

      this.renderList(this.pharmacies);
      this.initMap();
    } catch (err) {
      console.error(err);
      this.setStatus(t("api_error", this.lang), "danger");
    }
  }

  renderList(pharmacies) {
    if (!this.listEl) return;

    if (!pharmacies.length) {
      this.listEl.innerHTML = `<div class="text-center text-muted py-4">${t("no_pharmacies", this.lang)}</div>`;
      return;
    }

    this.listEl.innerHTML = pharmacies.map((p, i) => {
      const rating = p.rating ? `<span class="text-warning"><i class="bi bi-star-fill"></i> ${p.rating}</span>` : "";
      const phone = p.phone
        ? `<a href="tel:${p.phone}" class="small text-decoration-none"><i class="bi bi-telephone me-1"></i>${p.phone}</a>`
        : `<span class="small text-muted">${t("phone", this.lang)}: N/A</span>`;

      const statusClass = p.is_open === true ? "text-success" : p.is_open === false ? "text-danger" : "text-muted";
      const statusText = p.is_open === true ? t("open_now", this.lang)
        : p.is_open === false ? t("closed_now", this.lang)
        : t("hours_unknown", this.lang);

      return `
        <div class="glass-card pharmacy-card mb-3 p-3" data-index="${i}" role="button">
          <div class="d-flex justify-content-between align-items-start gap-2">
            <div>
              <h6 class="fw-bold mb-1">${this._esc(p.name)}</h6>
              <p class="small text-muted mb-1"><i class="bi bi-geo-alt me-1"></i>${this._esc(p.address)}</p>
              ${phone}
            </div>
            <div class="text-end flex-shrink-0">
              <span class="badge bg-primary-soft text-primary">${p.distance_km} km</span>
              <div class="small mt-1">${rating}</div>
            </div>
          </div>
          <div class="d-flex justify-content-between align-items-center mt-2 pt-2 border-top">
            <span class="small ${statusClass}"><i class="bi bi-clock me-1"></i>${statusText}</span>
            <a href="${p.maps_url}" target="_blank" rel="noopener"
               class="btn btn-sm btn-outline-primary rounded-pill"
               onclick="event.stopPropagation()">
              <i class="bi bi-map me-1"></i>${t("open_maps", this.lang)}
            </a>
          </div>
        </div>`;
    }).join("");

    this.listEl.querySelectorAll(".pharmacy-card").forEach(card => {
      card.addEventListener("click", () => {
        const idx = parseInt(card.dataset.index, 10);
        this.focusPharmacy(idx);
      });
    });
  }

  focusPharmacy(index) {
    const p = this.pharmacies[index];
    if (!p || !this.map) return;
    this.map.panTo({ lat: p.lat, lng: p.lng });
    this.map.setZoom(15);
    if (this.markers[index]) {
      google.maps.event.trigger(this.markers[index], "click");
    }
  }

  initMap() {
    if (!this.mapEl || !this.googleMapsKey || this.userLat == null) return;

    const loadMaps = () => {
      if (typeof google === "undefined" || !google.maps) {
        this.mapEl.innerHTML = `<p class="text-muted text-center py-5">${t("loading_map", this.lang)}</p>`;
        return;
      }

      const center = { lat: this.userLat, lng: this.userLng };
      this.map = new google.maps.Map(this.mapEl, {
        center,
        zoom: 13,
        mapTypeControl: false,
        streetViewControl: false,
        styles: [
          { featureType: "poi.medical", stylers: [{ visibility: "on" }] },
        ],
      });

      this.infoWindow = new google.maps.InfoWindow();
      this.markers.forEach(m => m.setMap(null));
      this.markers = [];

      // User marker
      new google.maps.Marker({
        position: center,
        map: this.map,
        title: "Your Location",
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 10,
          fillColor: "#1A73E8",
          fillOpacity: 1,
          strokeColor: "#fff",
          strokeWeight: 2,
        },
      });

      // Pharmacy markers
      this.pharmacies.forEach((p, i) => {
        const marker = new google.maps.Marker({
          position: { lat: p.lat, lng: p.lng },
          map: this.map,
          title: p.name,
          label: { text: String(i + 1), color: "white", fontSize: "11px", fontWeight: "bold" },
        });

        marker.addListener("click", () => {
          const content = `
            <div style="max-width:220px">
              <strong>${this._esc(p.name)}</strong><br>
              <small>${this._esc(p.address)}</small><br>
              <small>${p.distance_km} km · ${p.rating ? p.rating + "★" : "No rating"}</small><br>
              <a href="${p.maps_url}" target="_blank">${t("navigate", this.lang)}</a>
            </div>`;
          this.infoWindow.setContent(content);
          this.infoWindow.open(this.map, marker);
        });

        this.markers.push(marker);
      });
    };

    if (typeof google !== "undefined" && google.maps) {
      loadMaps();
    } else if (!window._mapsLoading) {
      window._mapsLoading = true;
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${this.googleMapsKey}&callback=_initPharmacyMap`;
      script.async = true;
      script.defer = true;
      window._initPharmacyMap = loadMaps;
      document.head.appendChild(script);
    } else {
      window._initPharmacyMap = loadMaps;
    }
  }

  _esc(str) {
    const d = document.createElement("div");
    d.textContent = str || "";
    return d.innerHTML;
  }
}
